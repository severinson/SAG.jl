"""

Stochastic average gradient method. Combining with a first-order optimization method results 
in a variance-reduced method for finite-sum optimization.
"""
module SAG

export StochasticGradient, update!, initialized_fraction

"""
    StochasticGradient{T,N} <: AbstractArray{T,N}

Stochastic average gradient (SAG) method for tracking a gradient made up of the sum of several 
sub-gradients. Combining SAG with an iterative method for first-order optimization method (e.g., 
gradient descent) results in a variance-reduced method for finite-sum optimization, e.g., empirical
risk minimization. Because it's a variance-reduced method, the iterate converges to the optimum 
even when the gradients used to update the iterate are stochastic.

External links:

* [Minimizing Finite Sums with the Stochastic Average Gradient](https://arxiv.org/abs/1309.2388)

```julia

# initialization
n = 3                                       # number of sub-gradients
dims = (10,)                                # gradient dimension
sg = StochasticGradient(Float32, n, dims)

# compute one sub-gradient at a time
i = 1                                       # sub-gradient index
∇i = randn(dims)                            # replace with actual sub-gradient computation
update!(sg, i, ∇i)                          # writes ∇i into the gradient
sg ≈ ∇i                                     # = true; sg acts as an array

# compute another sub-gradient
j = 2
∇j = randn(dims)                            # replace with actual sub-gradient computation
update!(sg, j, ∇j)
sg ≈ ∇i + ∇j                                # = true; sg acts as an array

# using StochasticGradient for variance-reduced stochastic gradient descent
w = randn(dims)                             # iterate, e.g., a machine learning model
stepsize = 0.1                              # step size used for model updates
f = initialized_fraction(sg)                # fraction of sub-gradients that have been initialized; 2/3 in our case
w .-= (stepsize / f) .* sg                  # gradient descent step

```

"""
mutable struct StochasticGradient{T,N} <: AbstractArray{T,N}
    ∇s::Vector{Array{T,N}}          # sub-gradients, or components
    ∇::Array{T,N}                   # gradient, i.e., the sum of ∇s
    isinitialized::Vector{Bool}     # indicates which sub-gradients that have been initialized
    ninitialized::Int               # number of sub-gradients that have been initialized        
end

function StochasticGradient(∇s::Vector{T}, ∇::T; overwrite=true) where T
    n = length(∇s)
    0 < n || throw(ArgumentError("n must be positive"))
    for i in 2:n
        size(∇s[i]) == size(∇s[1]) || throw(DimensionMismatch("The $(i)-th component has dimensions $(size(∇s[i])), but the first component has dimensions $(size(∇s[1]))"))
    end
    if overwrite
        ∇ .= ∇s[1]
        for i in 2:n
            ∇ .+= ∇s[i]
        end
    end
    StochasticGradient(∇s, ∇, zeros(Bool, n), 0)
end

function StochasticGradient(∇s::AbstractVector)
    length(∇s) > 0 || throw(ArgumentError("There must be at least 1 component"))
    StochasticGradient(∇s, similar(∇s[1]))
end

function StochasticGradient(∇::AbstractArray, n::Integer)
    0 < n || throw(ArgumentError("n must be positive"))
    ∇s = [similar(∇) for _ in 1:n]
    for ∇i in ∇s
        ∇i .= ∇ ./ n
    end
    StochasticGradient(∇s, ∇)
end

function StochasticGradient(T::DataType, n::Integer, dims::Tuple)
    0 < n || throw(ArgumentError("n must be positive"))
    StochasticGradient([zeros(T, dims) for _ in 1:n], zeros(T, dims))
end

StochasticGradient(n::Integer, dims::Tuple) = StochasticGradient(Float64, n, dims)

Base.length(sg::StochasticGradient) = length(sg.∇)
Base.size(sg::StochasticGradient, args...) = size(sg.∇, args...)
Base.getindex(sg::StochasticGradient, args...) = getindex(sg.∇, args...)
Base.view(sg::StochasticGradient, args...) = view(sg.∇, args...)
Base.eltype(sg::StochasticGradient) = eltype(sg.∇)
Base.iterate(sg::StochasticGradient, args...; kwargs...) = iterate(sg.∇, args...; kwargs...)
Base.similar(sg::StochasticGradient, args...) = StochasticGradient(similar(sg.∇, args...), length(sg.∇s))
Base.show(io::IO, sg::StochasticGradient) = write(io, "SAG{eltype: $(eltype(sg)), n: $(length(sg.∇s)), dims: $(size(sg.∇))}")

"""
    update!(sg::StochasticGradient, i::Integer, ∇i::AbstractArray)

Set the `i`-th component to `∇i`, and update the overall sum `∇` accordingly.
"""
function update!(sg::StochasticGradient, i::Integer, ∇i::AbstractArray)
    0 < i <= length(sg.∇s) || throw(ArgumentError("i is $i, but there are $(length(sg.∇s)) components"))
    size(∇i) == size(sg.∇) || throw(DimensionMismatch("∇i has dimensions $(size(∇i)), but ∇ has dimensions $(size(sg.∇))"))
    sg.∇ .+= ∇i .- sg.∇s[i]
    sg.∇s[i] .= ∇i
    if !sg.isinitialized[i]
        sg.ninitialized += 1
        sg.isinitialized[i] = true        
    end
    return ∇i
end

"""
    update!(sg::StochasticGradient, pair::Pair{<:Integer, <:AbstractArray})

"""
update!(sg::StochasticGradient, pair::Pair{<:Integer, <:AbstractArray}) = update!(sg, pair[1], pair[2])

"""
    update!(sg::StochasticGradient, pairs)

Perform an update for each element of `pairs`, which must be an iterable of collections of length 2
(e.g., pairs or tuples), where the second element is a component and the first is the index of that
component.
"""
function update!(sg::StochasticGradient, pairs)
    if 2*length(pairs) <= length(sg.∇s)
        for (i, ∇i) in pairs        
            update!(sg, i, ∇i)
        end
    else
        for (i, ∇i) in pairs
            sg.∇s[i] .= ∇i
            if !sg.isinitialized[i]
                sg.ninitialized += 1
                sg.isinitialized[i] = true                
            end
        end
        sg.∇ .= sg.∇s[1]
        for i in 2:length(sg.∇s)
            sg.∇ .+= sg.∇s[i]
        end
    end
    return pairs
end

initialized_fraction(::Any) = 1.0

"""

Return the fraction of components of `sg` that have been initialized.
"""
initialized_fraction(sg::StochasticGradient) = sg.ninitialized / length(sg.∇s)

end
