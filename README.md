# SAG

[Stochastic average gradient](https://arxiv.org/abs/1309.2388) (SAG) method for tracking a gradient made up of the sum of several sub-gradients. Combining SAG with an iterative method for first-order optimization method (e.g., gradient descent) results in a variance-reduced method for finite-sum optimization, e.g., empirical risk minimization. Because it's a variance-reduced method, the iterate converges to the optimum even when the gradients used to update the iterate are stochastic. See below for an example.

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