"""
GPU support for GenericFFT, via KernelAbstractions.

Loaded when both `GPUArraysCore` and `KernelAbstractions` are present, which is what any of
the vendor packages (CUDA, AMDGPU, Metal, oneAPI) pull in.

The design principle throughout: **every transcendental is evaluated on the host**. Twiddle
factors and Bluestein chirps are built in full generic precision with the same code the CPU
path uses, then uploaded once. The kernels below therefore need nothing from `T` beyond
complex `+`, `-` and `*`, which is what makes them compilable for a type such as
`DoubleFloats.Double64` that has no GPU-native intrinsics.
"""
module GenericFFTGPUExt

using GenericFFT
using GPUArraysCore: AbstractGPUArray
using KernelAbstractions
using KernelAbstractions: get_backend, synchronize

using GenericFFT: KAEngine, _twiddles, _like, _padded, generic_fft_pow2,
                  _batched_fft_first_dim!, _generic_fft_vec, _generic_fft_vec!,
                  copycomplex

# The single dispatch seam. Deliberately not `plan_fft(::AbstractGPUArray, region)`: a
# `CuArray` is both an `AbstractGPUArray` and a `StridedArray`, and neither type is a
# subtype of the other, so such a method would be ambiguous with the `StridedArray` plan
# methods in src/fft.jl rather than more specific than them.
function GenericFFT._engine(x::AbstractGPUArray{T}) where T
    isbitstype(T) || throw(ArgumentError(
        "GenericFFT cannot transform eltype $T on a GPU: it is not an isbits type. " *
        "BigFloat (MPFR) and Quadmath.Float128 are backed by C libraries and can never " *
        "run in a device kernel; try DoubleFloats.Double64 for extended precision."))
    return KAEngine(get_backend(x))
end

# ---------------------------------------------------------------------------
# Kernels. Arithmetic only -- no transcendentals, no branches on data.
# ---------------------------------------------------------------------------

# One stage of the radix-2 Stockham auto-sort transform. Every butterfly of a stage is
# independent, so the whole stage is a single launch over (half-length, stride, batch).
@kernel function stockham_stage!(dst, @Const(src), @Const(W), s::Int, h::Int, j::Int)
    p, r, q = @index(Global, NTuple)
    P = p - 1
    R = r - 1
    w = W[P*j + 1]
    u = src[R + s*P + 1, q]
    v = src[R + s*(P + h) + 1, q]
    dst[R + s*(2P) + 1, q]     = u + v
    dst[R + s*(2P + 1) + 1, q] = (u - v) * w
end

"""
    _stockham!(backend, dst, src, W) -> result

Batched power-of-two FFT over the columns of `src`, using `dst` as scratch. Returns
whichever buffer holds the result. Launches are queued on one backend stream and so run in
stage order without explicit synchronisation between them.
"""
function _stockham!(backend, dst::AbstractMatrix, src::AbstractMatrix, W::AbstractVector)
    n, nbatch = size(src)
    a, b = src, dst
    s = 1
    j = 1
    len = n
    kern = stockham_stage!(backend)
    while len > 1
        h = len >> 1
        kern(b, a, W, s, h, j; ndrange=(h, s, nbatch))
        a, b = b, a
        s <<= 1
        j <<= 1
        len = h
    end
    synchronize(backend)
    return a
end

# ---------------------------------------------------------------------------
# Batched transforms
# ---------------------------------------------------------------------------

function GenericFFT._batched_fft_first_dim!(e::KAEngine, y::AbstractMatrix{Complex{R}}) where {R<:AbstractFloat}
    n = size(y, 1)
    n <= 1 && return y
    return ispow2(n) ? _pow2!(e, y) : _bluestein!(e, y)
end

function _pow2!(e::KAEngine, y::AbstractMatrix{Complex{R}}) where {R<:AbstractFloat}
    W = _like(y, _twiddles(Complex{R}, size(y, 1)))
    res = _stockham!(e.backend, similar(y), y, W)
    res === y || copyto!(y, res)
    return y
end

"""
Batched Bluestein transform for non-power-of-two lengths.

The chirp `Wks` and the transformed filter `WQ` depend only on the length, never on the
data, so both are built once on the host -- `WQ` by the existing CPU power-of-two kernel --
and shared across the whole batch.
"""
function _bluestein!(e::KAEngine, y::AbstractMatrix{Complex{R}}) where {R<:AbstractFloat}
    n, nbatch = size(y)
    S = promote_type(R, Float64)          # working precision, matching the CPU path

    ks = range(zero(S), stop=S(n)-one(S), length=n)
    Wks_h = Complex{R}.(cispi.(-ks.^2 ./ S(n)))
    wq_h = conj!([Complex{R}(cispi(-S(n))); Wks_h[end:-1:1]; Wks_h[2:end]])

    nconv = 3n - 1                        # length(xq) + length(wq) - 1
    np2 = nextpow(2, nconv)

    # Bluestein convolves in a working precision of at least Float64, matching the CPU
    # path exactly. A backend without Float64 (Metal) therefore cannot run non-power-of-two
    # lengths for the narrower eltypes; say so plainly rather than let a vendor-level error
    # surface from deep inside the buffer allocation.
    if S !== R
        try
            similar(y, Complex{S}, 1)
        catch
            throw(ArgumentError(
                "GenericFFT: a length-$n (non-power-of-two) transform of eltype " *
                "Complex{$R} needs Complex{$S} working buffers, which this backend does " *
                "not support. Pad the transform to a power of two, or use an eltype of " *
                "at least Float64 such as Float64 or DoubleFloats.Double64."))
        end
    end

    Wks  = _like(y, Wks_h)
    WQ   = _like(y, generic_fft_pow2(_padded(wq_h, Complex{S}, np2)))
    Wtab = _like(y, _twiddles(Complex{S}, np2))

    buf = similar(y, Complex{S}, np2, nbatch)
    fill!(buf, zero(Complex{S}))
    @views buf[1:n, :] .= y .* Wks
    scratch = similar(buf)

    fwd = _stockham!(e.backend, scratch, buf, Wtab)
    fwd .*= WQ

    # Inverse transform of length np2: conjugate, forward, conjugate, scale.
    fwd .= conj.(fwd)
    other = fwd === buf ? scratch : buf
    res = _stockham!(e.backend, other, fwd, Wtab)
    res .= conj.(res) ./ np2

    @views y .= Wks .* res[n+1:2n, :]
    return y
end

# Vector entry points: a length-n vector is just a batch of one.
GenericFFT._generic_fft_vec!(e::KAEngine, x::AbstractVector) =
    (_batched_fft_first_dim!(e, reshape(x, :, 1)); x)

GenericFFT._generic_fft_vec(e::KAEngine, x::AbstractVector) =
    _generic_fft_vec!(e, copycomplex(x))

end # module
