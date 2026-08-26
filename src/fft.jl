# We use these type definitions for clarity
const RealFloats = T where T<:AbstractFloat
const ComplexFloats = Complex{T} where T<:AbstractFloat
const AbstractFloats = Union{RealFloats, ComplexFloats}

# The following implements Bluestein's algorithm, following http://www.dsprelated.com/dspbooks/mdft/Bluestein_s_FFT_Algorithm.html

function generic_fft!(x::AbstractVector{Complex{T}}) where {T<:AbstractFloat}
    if ispow2(length(x))
        return generic_fft_pow2!(x)
    end
    return copyto!(x, generic_fft(x))
end

generic_fft!(x::AbstractVector) = copyto!(x, generic_fft(x))

function generic_fft!(x::AbstractVector{Complex{T}}, region::Integer) where {T<:AbstractFloat}
    @assert region == 1
    generic_fft!(x)
end

# A batch of 1-D transforms along the first dimension. In column-major order the trailing
# dimensions are just a flat batch axis, so a single `reshape` (which never copies) exposes
# the batch without any `CartesianIndices` bookkeeping.
function _generic_fft_first_dim!(x)
    n = size(x, 1)
    _batched_fft_first_dim!(reshape(x, n, :))
    x
end

# Batches smaller than this are not worth the task-spawn overhead.
const THREAD_MIN_BATCH = 16

function _batched_fft_first_dim!(y::AbstractMatrix)
    # The slices are independent, so the batch axis parallelises directly. Tasks inherit
    # the BigFloat precision of the spawning task, so an enclosing `setprecision` block
    # still applies inside the loop.
    if Threads.nthreads() > 1 && size(y, 2) >= THREAD_MIN_BATCH
        Threads.@threads for j in axes(y, 2)
            generic_fft!(@view y[:, j])
        end
    else
        for j in axes(y, 2)
            generic_fft!(@view y[:, j])
        end
    end
    y
end

function generic_fft!(x, region::Integer)
    @assert 1 <= region <= ndims(x)

    # Dimension 1 is already contiguous, so transform in place where it lies. Permuting
    # here would be two full-array copies for an identity permutation.
    region == 1 && return _generic_fft_first_dim!(x)

    perm = ntuple(ndims(x)) do i
        if i == 1
            region
        elseif i == region
            1
        else
            i
        end
    end

    # For region > 1 we bring the transformed dimension to the front rather than striding
    # over it: the 1-D kernels are scalar loops, and they want contiguous input.
    y = permutedims(x, perm)
    _generic_fft_first_dim!(y)
    permutedims!(x, y, perm)
    x
end

function generic_fft!(x, region)
    for r in region
        generic_fft!(x, r)
    end
    x
end

function generic_fft!(x)
    y = similar(x)
    z = x
    sz = size(x)
    perm = ((2:ndims(x))..., 1)

    for r in 1:ndims(x)
        _generic_fft_first_dim!(z)

        sz = (sz[2:end]..., sz[1])
        y = reshape(y, sz)
        permutedims!(y, z, perm)
        z, y = y, z
    end

    if isodd(ndims(x))
        x .= z
    end
    x
end


# generic_fft(x, region) = generic_fft!(copy(complex(x)), region)
# generic_fft(x) = generic_fft!(copy(complex(x)))

# Replace entry `d` of a size tuple. Unlike `collect(size(x))` + `tuple(sz...)` this keeps
# the tuple length inferable, so the `similar` calls below stay type stable.
_setindex(sz::NTuple{N,Int}, val::Int, d::Integer) where N = ntuple(i -> i == d ? val : sz[i], N)

copycomplex(A::AbstractArray{<:Complex}) = copy(A)
copycomplex(A::AbstractArray{<:Real}) = complex(A)
generic_fft(x, region) = generic_fft!(copycomplex(x), region)
generic_fft(x) = generic_fft!(copycomplex(x))


function generic_fft(x::AbstractVector{T}) where T<:AbstractFloats
    n = length(x)
    ispow2(n) && return generic_fft_pow2(x)
    S = promote_type(real(T), Float64)
    ks = range(zero(S), stop=S(n)-one(S), length=n)
    Wks = Complex{real(T)}.(cispi.(-ks.^2 ./ S(n)))    # always Complex
    Wksrev = @view Wks[reverse(eachindex(Wks))]
    xq, wq = complex(x).*Wks, conj!([Complex{real(T)}(cispi(-S(n))); Wksrev; @view Wks[2:end]])
    return Wks .* @view _conv(xq,wq)[n+1:2n]
end

generic_bfft(x::AbstractArray{T, N}, region) where {T <: AbstractFloats, N} = conj!(generic_fft(conj(x), region))
generic_bfft!(x::AbstractArray{T, N}, region) where {T <: AbstractFloats, N} = conj!(generic_fft!(conj!(x), region))

_regionscale(x, region::Int) = size(x, region)
_regionscale(x, region) = prod(size.(Ref(x), region))

generic_ifft(x::AbstractArray{T, N}, region) where {T<:AbstractFloats, N} = ldiv!(T(_regionscale(x, region)), conj!(generic_fft(conj(x), region)))
generic_ifft!(x::AbstractArray{T, N}, region) where {T<:AbstractFloats, N} = ldiv!(T(_regionscale(x, region)), conj!(generic_fft!(conj!(x), region)))

generic_rfft(v::AbstractVector{T}, region) where T<:AbstractFloats = generic_fft(v, region)[1:div(length(v),2)+1]

function generic_rfft(x::AbstractArray{T, N}, region) where {T<:AbstractFloats, N}
    d = first(region)
    if length(region) > 1
        return generic_fft(generic_rfft(x, d), region[2:end])
    end

    nout = size(x, d) ÷ 2 + 1
    out = similar(x, Complex{real(T)}, _setindex(size(x), nout, d))

    # CartesianIndices enables iterating over slices in arbitrary dimensions
    Rpre = CartesianIndices(size(x)[1:d-1])
    Rpost = CartesianIndices(size(x)[d+1:end])

    for Ipost in Rpost
        for Ipre in Rpre
            out[Ipre, :, Ipost] .= generic_rfft(view(x, Ipre, :, Ipost), 1)
        end
    end
    return out
end

function generic_irfft(v::AbstractVector{T}, n::Integer, region) where T<:ComplexFloats
    m = n>>1 + 1
    @assert length(v) == m
    # `similar` rather than `Vector{T}(undef, n)`, so the buffer follows `v`'s array type.
    r = similar(v, n)
    copyto!(r, 1, v, 1, m)
    # Hermitian extension of the second half. Written as a reverse-strided broadcast
    # instead of a scalar loop so it stays a single vectorized pass.
    k = n - m
    k > 0 && @views r[m+1:n] .= conj.(v[k+1:-1:2])
    return real(generic_ifft(r, region))
end

function generic_irfft(x::AbstractArray{T, N}, n::Integer, region) where {T<:ComplexFloats, N}
    d = first(region)
    if length(region) > 1
        return generic_irfft(generic_ifft(x, region[2:end]), n, d)
    end

    out = similar(x, real(T), _setindex(size(x), Int(n), d))

    Rpre = CartesianIndices(size(x)[1:d-1])
    Rpost = CartesianIndices(size(x)[d+1:end])

    for Ipost in Rpost
        for Ipre in Rpre
            out[Ipre, :, Ipost] .= generic_irfft(view(x, Ipre, :, Ipost), n, 1)
        end
    end
    return out
end

function generic_brfft(v::AbstractArray, n::Integer, region)
    scale = n * _regionscale(v, region isa Integer ? () : region[2:end])
    return generic_irfft(v, n, region) * scale
end

"""
    _padded(x, ::Type{U}, n)

Copy `x` into a freshly allocated length-`n` array of eltype `U`, zeroing the tail.

Allocated with `similar(x, ...)` so the result follows the input's array type, and written
without resizing so `x` may be a view or a non-resizable (e.g. GPU) array.
"""
function _padded(x::AbstractVector, ::Type{U}, n::Integer) where U
    nx = length(x)
    y = similar(x, U, n)
    copyto!(y, 1, x, 1, nx)
    nx < n && fill!(view(y, nx+1:n), zero(U))
    return y
end

function _conv(u::AbstractVector{T}, v::AbstractVector{T}) where T<:AbstractFloats
    nu, nv = length(u), length(v)
    n  = nu + nv - 1
    np2 = nextpow(2, n)
    S = promote_type(real(T), Float64)
    # Zero-pad into new buffers rather than `append!`-ing the inputs: this leaves `u` and
    # `v` unmutated (so views are allowed) and never needs a resizable array. The padded
    # copies are free of charge, since the eltype promotion to `Complex{S}` allocated anyway.
    uf = _padded(u, Complex{S}, np2)
    vf = _padded(v, Complex{S}, np2)
    y = generic_ifft_pow2(generic_fft_pow2(uf) .* generic_fft_pow2(vf))
    return T <: Real ? T.(real(@view y[1:n])) : T.(@view y[1:n])
end


# This is a Cooley-Tukey FFT algorithm inspired by many widely available algorithms including:
# c_radix2.c in the GNU Scientific Library and four1 in the Numerical Recipes in C.
# However, the trigonometric recurrence is improved for greater efficiency.
# The algorithm starts with bit-reversal, then divides and conquers in-place.
function generic_fft_pow2!(x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    n,big2=2length(x),2one(T)
    nn,j=n÷2,1
    for i=1:nn
        if j>i
            x[j], x[i] = x[i], x[j]
        end
        m = nn÷2
        while m ≥ 2 && j > m
            j -= m
            m = m÷2
        end
        j += m
    end
    logn = 2
    while logn < n
        θ=-big2/logn
        wp = complex(-2sinpi(θ/2)^2, sinpi(θ))
        w = complex(one(T))
        lognn = logn ÷ 2
        for m=1:lognn
            for i=m:logn:nn
                j=i+lognn
                mix = w * x[j]
                x[j] = x[i] - mix
                x[i] = x[i] + mix
            end
            w = w * (1 + wp)
        end
        logn = 2logn
    end
    return x
end

function generic_fft_pow2(x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    return generic_fft_pow2!(copy(x))
end
generic_fft_pow2(x::AbstractVector{T}) where T<:AbstractFloat = generic_fft_pow2(complex(x))

function generic_ifft_pow2(x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    y = conj.(x)  # always create copy (conj(x) doesn't copy when eltype(x) is real)
    generic_fft_pow2!(y)
    N = T(length(x))
    @. y = conj(y) / N
    return y
end

function generic_dct(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    generic_dct(x)
end

function generic_dct!(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    copyto!(x, generic_dct(x))
end

function generic_idct(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    generic_idct(x)
end

function generic_idct!(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    @assert region == 1
    copyto!(x, generic_idct(x))
end

function generic_dct(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    generic_dct(x)
end

function generic_dct!(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    copyto!(x, generic_dct(x))
end

function generic_idct(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    generic_idct(x)
end

function generic_idct!(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    @assert region == 1:1
    copyto!(x, generic_idct(x))
end

function generic_dct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    T <: FFTW.fftwNumber && (@warn("Using generic dct for FFTW number type."))
    N = length(a)
    twoN = convert(T,2) * N
    c = generic_fft([a; reverse(a, dims=1)]) # c = generic_fft([a; flipdim(a,1)])
    d = c[1:N]
    d .*= exp.((-im*convert(T, pi)).*(0:N-1)./twoN)
    d[1] = d[1] / sqrt(convert(T, 2))
    lmul!(inv(sqrt(twoN)), d)
end

generic_dct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_dct(complex(a)))

function generic_idct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    T <: FFTW.fftwNumber && (@warn("Using generic idct for FFTW number type."))
    N = length(a)
    twoN = convert(T,2)*N
    b = a * sqrt(twoN)
    b[1] = b[1] * sqrt(convert(T,2))
    shift = exp.(-im * 2 * convert(T, pi) * (N - convert(T,1)/2) * (0:(2N-1)) / twoN)
    b = [b; 0; -reverse(b[2:end], dims=1)] .* shift # b = [b; 0; -flipdim(b[2:end],1)] .* shift
    c = ifft(b)
    reverse(c[1:N]; dims=1)#flipdim(c[1:N],1)
end

generic_idct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_idct(complex(a)))


# These lines mimick the corresponding ones in FFTW/src/dct.jl, but with
# AbstractFloat rather than fftwNumber.
for f in (:dct, :dct!, :idct, :idct!)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::AbstractArray{<:AbstractFloats}) = $pf(x) * x
        $f(x::AbstractArray{<:AbstractFloats}, region) = $pf(x, region) * x
    end
end

# dummy plans
abstract type DummyPlan{T} <: Plan{T} end
for P in (:DummyFFTPlan, :DummyiFFTPlan, :DummybFFTPlan, :DummyDCTPlan, :DummyiDCTPlan)
    # All plans need an initially undefined pinv field
    @eval begin
        mutable struct $P{T,inplace,G} <: DummyPlan{T}
            region::G # region (iterable) of dims that are transformed
            pinv::Plan
            $P{T,inplace,G}(region::G) where {T<:AbstractFloats, inplace, G} = new(region)
        end
    end
end
for P in (:DummyrFFTPlan, :DummyirFFTPlan, :DummybrFFTPlan)
    @eval begin
        mutable struct $P{T,inplace,G} <: DummyPlan{T}
            n::Integer
            region::G
            pinv::Plan
            $P{T,inplace,G}(n::Integer, region::G) where {T<:AbstractFloats, inplace, G} = new(n, region)
        end
    end
end

for (Plan,iPlan) in ((:DummyFFTPlan,:DummyiFFTPlan),
                     (:DummyDCTPlan,:DummyiDCTPlan))
   @eval begin
       plan_inv(p::$Plan{T,inplace,G}) where {T,inplace,G} = $iPlan{T,inplace,G}(p.region)
       plan_inv(p::$iPlan{T,inplace,G}) where {T,inplace,G} = $Plan{T,inplace,G}(p.region)
    end
end

# Specific for rfft, irfft and brfft:
plan_inv(p::DummyirFFTPlan{T,inplace,G}) where {T,inplace,G} = DummyrFFTPlan{real(T),inplace,G}(p.n, p.region)
plan_inv(p::DummyrFFTPlan{T,inplace,G}) where {T,inplace,G} = DummyirFFTPlan{Complex{T},inplace,G}(p.n, p.region)



# The complex and trigonometric transforms have both an in-place and an out-of-place form.
for (Plan,ff,ff!) in ((:DummyFFTPlan,:generic_fft,:generic_fft!),
                      (:DummybFFTPlan,:generic_bfft,:generic_bfft!),
                      (:DummyiFFTPlan,:generic_ifft,:generic_ifft!),
                      (:DummyDCTPlan,:generic_dct,:generic_dct!),
                      (:DummyiDCTPlan,:generic_idct,:generic_idct!))
    @eval begin
        *(p::$Plan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff!(x, p.region)
        *(p::$Plan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff(x, p.region)
        function mul!(C::StridedVector, p::$Plan, x::StridedVector)
            C[:] = $ff(x, p.region)
            C
        end
    end
end

# The real transforms (rfft, irfft, brfft) have no in-place form: they change the length of
# the transformed dimension, so the result cannot alias the input. Accordingly `plan_rfft`,
# `plan_irfft` and `plan_brfft` below all hard-code `inplace=false`, and there is
# deliberately no `inplace=true` method here.
for (Plan,ff) in ((:DummyrFFTPlan,:generic_rfft),)
    @eval begin
        *(p::$Plan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff(x, p.region)
        function mul!(C::StridedVector, p::$Plan, x::StridedVector)
            C[:] = $ff(x, p.region)
            C
        end
    end
end

for (Plan,ff) in ((:DummyirFFTPlan,:generic_irfft), (:DummybrFFTPlan,:generic_brfft))
    @eval begin
        *(p::$Plan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff(x, p.n, p.region)
        function mul!(C::StridedVector, p::$Plan, x::StridedVector)
            C[:] = $ff(x, p.n, p.region)
            C
        end
    end
end


# We intercept the calls to plan_X(x, region) below.
# In order not to capture any calls that should go to FFTW, we have to be
# careful about the typing, so that the calls to FFTW remain more specific.
# This is the reason for using StridedArray below. We also have to carefully
# distinguish between real and complex arguments.

plan_fft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyFFTPlan{T,false,typeof(region)}(region)
plan_fft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyFFTPlan{T,true,typeof(region)}(region)
plan_fft(x::StridedArray{T}, region; kws...) where {T <: RealFloats} =
    T <: FFTW.fftwReal ? invoke(plan_fft, Tuple{AbstractArray{<:Real}, Any}, x, region; kws...) : DummyFFTPlan{Complex{T},false,typeof(region)}(region)
plan_fft!(x::StridedArray{T}, region; kws...) where {T <: RealFloats} =
    T <: FFTW.fftwReal ? invoke(plan_fft!, Tuple{AbstractArray, Any}, x, region; kws...) : DummyFFTPlan{Complex{T},true,typeof(region)}(region)

# intercept fft(x) before AbstractFFTs gets a chance for any non-FFTW float type.
fft(x::StridedArray{T}) where {T<:AbstractFloats} = generic_fft(x)
fft(x::StridedArray{T}, region) where {T<:AbstractFloats} = generic_fft(x, region)

plan_bfft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummybFFTPlan{T,false,typeof(region)}(region)
plan_bfft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummybFFTPlan{T,true,typeof(region)}(region)

# The ifft plans are automatically provided in terms of the bfft plans above.
# plan_ifft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyiFFTPlan{Complex{real(T)},false,typeof(region)}(region)
# plan_ifft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyiFFTPlan{Complex{real(T)},true,typeof(region)}(region)

plan_dct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,false,typeof(region)}(region)
plan_dct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,true,typeof(region)}(region)

plan_idct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,false,typeof(region)}(region)
plan_idct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,true,typeof(region)}(region)

plan_rfft(x::StridedArray{T}, region) where {T <: RealFloats} = DummyrFFTPlan{T,false,typeof(region)}(size(x, first(region)), region)
plan_brfft(x::StridedArray{T}, n::Integer, region) where {T <: ComplexFloats} = DummybrFFTPlan{T,false,typeof(region)}(n, region)

# Explicitly define plan_irfft to ensure correct scaling
plan_irfft(x::StridedArray{T}, n::Integer, region) where {T <: ComplexFloats} = DummyirFFTPlan{T,false,typeof(region)}(n, region)

# These don't exist for now:
# plan_rfft!(x::StridedArray{T}) where {T <: RealFloats} = DummyrFFTPlan{Complex{real(T)},true}()
# plan_irfft!(x::StridedArray{T},n::Integer) where {T <: RealFloats} = DummyirFFTPlan{Complex{real(T)},true}()

# old version deprecated in favour of interlace_complex, deinterlace_complex below
function interlace(a::AbstractVector{S},b::AbstractVector{V}) where {S<:Number,V<:Number}
    na=length(a);nb=length(b)
    T=promote_type(S,V)
    if nb≥na
        ret=zeros(T,2nb)
        ret[1:2:1+2*(na-1)]=a
        ret[2:2:end]=b
        ret
    else
        ret=zeros(T,2na-1)
        ret[1:2:end]=a
        if !isempty(b)
            ret[2:2:2+2*(nb-1)]=b
        end
        ret
    end
end

"""
Interlace `a::AbstractVector` with complex entries as

    [(r1,i1), (r2,i2), (r2,i3), ...] -> [r1,i1,r2,i2,r3,i3,...]
    
with `r,i` the real and imaginary part of every element in `a`."""
function interlace_complex(a::AbstractVector{S}, conjfn = identity) where {S<:Complex}
    n = length(a)
    a_interlaced = zeros(real(S),2n)
    @inbounds for (i, ai) in enumerate(a)
        a_interlaced[2i-1] = real(ai)
        a_interlaced[2i] = conjfn(imag(ai))
    end
    return a_interlaced
end

"""Reverse function of interlace_complex."""
function deinterlace_complex(a_interlaced::AbstractVector{S}, conjfn = identity) where {S<:Real}
    n = length(a_interlaced)
    a = zeros(Complex{S}, n ÷ 2)        # deinterlaced vector
    @inbounds for i in eachindex(a)     # ignores last entry if length(a_interlaced) odd
        a[i] = complex(a_interlaced[2i-1], conjfn(a_interlaced[2i]))
    end
    return a
end
