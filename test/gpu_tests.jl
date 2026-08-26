# Tests for the GenericFFTGPUExt extension.
#
# The KernelAbstractions kernels are backend-generic, so running them over the `CPU()`
# backend executes exactly the code a GPU would run -- same kernels, same host-side table
# construction, same buffer juggling -- without requiring the hardware. That is what makes
# this suite meaningful in ordinary CI. A run on real devices additionally needs the
# relevant vendor package, which is out of scope for the test target here.

using GenericFFT, Test, DoubleFloats
using GPUArraysCore, KernelAbstractions
using GenericFFT: KAEngine, _batched_fft_first_dim!, _generic_fft_vec, _generic_fft_vec!,
                  generic_fft, generic_ifft

const KAE = KAEngine(CPU())

# A minimal AbstractGPUArray stand-in for the eltype gate: `_engine` checks the eltype
# before it ever touches the array, so no array interface methods are needed. Declared at
# top level because struct definitions are not permitted in local scope.
struct FakeGPUArray{T,N} <: AbstractGPUArray{T,N}
    data::Array{T,N}
end

# Column-by-column reference from the (independently tested) CPU implementation.
refcols(X) = reduce(hcat, [generic_fft(X[:, q]) for q in 1:size(X, 2)])

relerr(got, ref) = Float64(maximum(abs, got .- ref) / max(maximum(abs, ref), 1))

@testset "GPU extension" begin
    @test Base.get_extension(GenericFFT, :GenericFFTGPUExt) !== nothing

    @testset "batched transform matches CPU — $T" for T in (Float64, Double64)
        # Powers of two take the Stockham path; the rest take batched Bluestein.
        for n in (1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 31, 32, 64), nbatch in (1, 3, 8)
            X = randn(Complex{T}, n, nbatch)
            Y = copy(X)
            _batched_fft_first_dim!(KAE, Y)
            @test relerr(Y, refcols(X)) < 1000*eps(T)
        end
    end

    @testset "vector entry points — $T" for T in (Float64, Double64)
        for n in (1, 2, 6, 8, 15, 16)
            x = randn(Complex{T}, n)
            @test relerr(_generic_fft_vec(KAE, x), generic_fft(x)) < 1000*eps(T)

            # in-place form must leave the result in the argument
            y = copy(x)
            out = _generic_fft_vec!(KAE, y)
            @test out === y
            @test relerr(y, generic_fft(x)) < 1000*eps(T)

            # real input is complexified rather than rejected
            xr = randn(T, n)
            @test relerr(_generic_fft_vec(KAE, xr), generic_fft(xr)) < 1000*eps(T)
        end
    end

    @testset "round trip — $T" for T in (Float64, Double64)
        for n in (4, 6, 16, 20)
            X = randn(Complex{T}, n, 4)
            Y = copy(X)
            _batched_fft_first_dim!(KAE, Y)
            # inverse via conjugation, the same identity generic_ifft uses
            Z = conj.(Y)
            _batched_fft_first_dim!(KAE, Z)
            Z = conj.(Z) ./ n
            @test relerr(Z, X) < 1000*eps(T)
        end
    end

    @testset "does not disturb the input's own buffer" begin
        # _bluestein! writes the result back into `y`; make sure it returns `y` itself and
        # not one of its scratch buffers, for both an odd and an even non-power-of-two.
        for n in (6, 7)
            Y = randn(ComplexF64, n, 2)
            @test _batched_fft_first_dim!(KAE, Y) === Y
        end
    end

    @testset "unsupported eltypes are rejected with a usable message" begin
        err = try
            GenericFFT._engine(FakeGPUArray(zeros(BigFloat, 4)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("isbits", err.msg)
        @test occursin("Double64", err.msg)
    end
end
