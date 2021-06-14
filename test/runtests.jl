using SAG
using Test

@testset "SAG.jl" begin
    # Test constructors
    dims = (3, 2)
    n = 3
    ∇s = [randn(dims) for _ in 1:n]
    sg = StochasticGradient(∇s, zeros(dims), overwrite=false)
    @test sg ≈ zeros(dims)
    @test sg ≈ sg.∇
    sg = StochasticGradient(∇s, zeros(dims))
    @test sg ≈ sum(∇s)
    sg = StochasticGradient(∇s)
    @test sg ≈ sum(∇s)
    ∇ = randn(dims)    
    sg = StochasticGradient(copy(∇), n)
    @test sg ≈ ∇
    @test sum(sg.∇s) ≈ ∇
    sg = StochasticGradient(n, dims)
    @test sg ≈ zeros(dims)
    @test sum(sg.∇s) ≈ zeros(dims)
    sg = StochasticGradient(Int, n, dims)  
    @test sg ≈ zeros(dims)
    @test sum(sg.∇s) ≈ zeros(dims)
    @test eltype(sg) == Int
    @test eltype(sg.∇) == Int
    @test eltype(sg.∇s[1]) == Int

    # Test updates
    dims = (3, 2)
    n = 3
    i = 1
    ∇i = randn(dims)    
    sg = StochasticGradient(n, dims)
    update!(sg, i, ∇i)
    @test sg.∇s[i] ≈ ∇i
    @test sg ≈ ∇i
    @test initialized_fraction(sg) ≈ 1/3
    sg = StochasticGradient(n, dims)
    update!(sg, i=>∇i)
    @test sg.∇s[i] ≈ ∇i
    @test sg ≈ ∇i
    @test initialized_fraction(sg) ≈ 1/3
    sg = StochasticGradient(n, dims)
    update!(sg, [1=>∇i, 2=>∇i])
    @test sg.∇s[1] ≈ ∇i
    @test sg.∇s[2] ≈ ∇i
    @test sg ≈ 2∇i
    @test initialized_fraction(sg) ≈ 2/3
    sg = StochasticGradient(n, dims)
    update!(sg, zip([1,2], [∇i, ∇i]))
    @test sg.∇s[1] ≈ ∇i
    @test sg.∇s[2] ≈ ∇i
    @test sg ≈ 2∇i
    @test initialized_fraction(sg) ≈ 2/3
end
