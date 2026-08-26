# ---------------------------------------------------------------------------
# Execution engines
#
# Every array-backend-specific decision funnels through `_engine`. The GPU extension
# overloads exactly this one function and the batched kernel it selects.
#
# Dispatching plan construction on the array type instead would be ambiguous: a `CuArray`
# is both an `AbstractGPUArray` and a `StridedArray`, and neither is a subtype of the
# other, so `plan_fft(::AbstractGPUArray, region)` would clash with the `StridedArray`
# methods in fft.jl.
# ---------------------------------------------------------------------------

abstract type FFTEngine end

struct CPUEngine <: FFTEngine end

"""
    KAEngine(backend)

Run the transforms as KernelAbstractions kernels on `backend`.

Selected by `_engine` for GPU arrays; the kernels themselves are backend-generic, so
constructing one over `KernelAbstractions.CPU()` runs the identical code paths on the host,
which is how they are covered by CI without GPU hardware. The methods live in the
`GenericFFTGPUExt` extension; only the type is declared here so it can be named without it.
"""
struct KAEngine{B} <: FFTEngine
    backend::B
end

_engine(::AbstractArray) = CPUEngine()

# `conj!` by broadcast. Base's `conj!(::AbstractArray)` is a scalar-indexed loop, which
# array types that disallow scalar indexing reject outright.
_conj!(A::AbstractArray) = (A .= conj.(A); A)

"""
    _twiddles(Complex{T}, n) -> Vector{Complex{T}}

The `n÷2` roots of unity `exp(-2πik/n)`, `k = 0:n÷2-1`, for a length-`n` transform.

A single table serves every stage of a radix-2 transform: the stage working on length
`n >> j` reads it with stride `2^j`. Computed on the host, in the same working precision
Bluestein's algorithm uses, so that the transform kernels themselves need nothing beyond
complex `+`, `-` and `*` — which is what makes them compilable for an exotic `T` on a GPU.
"""
function _twiddles(::Type{Complex{T}}, n::Integer) where T<:AbstractFloat
    m = max(n >> 1, 1)
    S = promote_type(T, Float64)
    W = Vector{Complex{T}}(undef, m)
    @inbounds for k in 0:m-1
        W[k+1] = Complex{T}(cispi(-2*S(k)/S(n)))
    end
    return W
end

"""
    _fft_pow2_batched!(engine, dst, src, W) -> result

Batched radix-2 Stockham auto-sort FFT over the columns of `src`, a `(n, batch)` matrix
with `n` a power of two. `dst` is scratch of the same size, `W` the table from
[`_twiddles`](@ref). Returns whichever of `src`/`dst` holds the result, in natural order.

Stockham rather than the in-place Cooley-Tukey of `generic_fft_pow2!`: it needs no
bit-reversal permutation and carries no state between butterflies, so every butterfly of a
stage is independent and the whole stage maps to one parallel launch. The twiddles are
read from `W` by index rather than built by the recurrence `w *= (1 + wp)`, which is both
parallelisable and more accurate, the recurrence being a compounding approximation.
"""
function _fft_pow2_batched!(::CPUEngine, dst::AbstractMatrix, src::AbstractMatrix, W::AbstractVector)
    n, nbatch = size(src)
    a, b = src, dst
    s = 1          # stride between the two halves of a butterfly
    j = 1          # stride into the twiddle table, always n ÷ len
    len = n
    while len > 1
        h = len >> 1
        @inbounds for q in 1:nbatch
            for p in 0:h-1
                w = W[p*j + 1]
                for r in 0:s-1
                    u = a[r + s*p + 1, q]
                    v = a[r + s*(p + h) + 1, q]
                    b[r + s*(2p) + 1, q]     = u + v
                    b[r + s*(2p + 1) + 1, q] = (u - v) * w
                end
            end
        end
        a, b = b, a
        s <<= 1
        j <<= 1
        len = h
    end
    return a
end
