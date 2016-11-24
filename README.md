# *rcss* - R Package

## Authors
Juri Hinz and Jeremy Yee

## Description

R package for the numerical treatment of convex switching
systems. Requires the *rflann* package (also found on my github
account). Please contact me by email (jeremyyee@outlook.com.au) or
through my github account to report any issues.

An implementation in the *Julia* language is also available on my
GitHub page.

## Problem Setting

A convex switching system is basically a Markov decision process with:
* a finite number of time points
* a Markov process consisting of:
      1. a controlled Markov chain with a finite number of possible realizations (positions) 
      2. a continuous process that evolves linearly i.e. **X<sub>t+1</sub> = W<sub>t+1</sub> X<sub>t</sub>**
      where **W<sub>t+1</sub>** is a matrix with random entries
* reward functions that are convex and Lipschitz continuous in the continuous 
  process
* a finite number of actions

Using this R package, we can then approximate all the value functions
in the Bellman recursion and also compute their lower and upper bounds.
The following R code demonstrates this.

## Example: Bermuda Put Option - Value Function Approximation

Let us consider the valuation of a Bermuda put option with strike
price **40** and expiry date of **1** year. The underlying asset price
follows a geometric Brownian motion. We assume the option is
exercisable at 51 evenly spaced time points, including one at
beginning and one at the end of the year. We first set our parameters.

~~~
# Parameters
rate <- 0.06 ## Interest rate
step <- 0.02 ## Time step between decision epochs
vol <- 0.2 ## Volatility of stock price process
n_dec <- 51 ## Number of decision epochs
~~~

The following then sets the transition probabilities for the controlled Markov
chain.

~~~
# Stochastic position control
control <- array(data = 0, dim = c(2,2,2))
control[2,1,2] <- 1
control[2,2,1] <- 1
control[1,1,1] <- 1
control[1,2,1] <- 1
~~~

Alternatively, for deterministic position control, a matrix can be suppplied.
This method is also quicker.

~~~
# Deterministic position control
control <- matrix(c(c(1, 1), c(2, 1)), nrow = 2, byrow = TRUE)
~~~

Next, we define an equally spaced grid ranging from 10 to 100 and
comprising 181 grid points.

~~~
# Grid
n_grid <- 181
grid <- as.matrix(cbind(rep(1, n_grid), seq(10, 100, length = n_grid)))
~~~

Introduce the reward functions in terms of sub-gradients:

~~~
# Subgradient representation of reward
strike <- 40
in_money <- grid[,2] <= strike
reward <- array(0, dim = c(n_grid, 2, 2, 2, n_dec))       
reward[in_money, 1, 2, 2,] <- strike
reward[in_money, 2, 2, 2,] <- -1
for (i in 1:n_dec){
    reward[,,,,i] <- exp(-rate * step * (i - 1)) * reward[,,,,i] 
}
~~~

where **reward[1, ,p, a, t]** represents the intercept and 
**reward[, ,p, a, t]** gives the slope associated with position
**p**, action **a** and time **t**. Finally, we define the sampling of 
disturbances **(W<sub>t</sub>)** which the code assumes tho be indentically
distributed across time.

~~~
# Disturbance  discretization
n_disturb <- 10000 # number of disturbances
weight <- rep(1/n_disturb, n_disturb) # neights
disturb <- array(0, dim = c(2, 2, n_disturb))
disturb[1, 1,] <- 1
quantile <- qnorm(seq(0, 1, length = (n_disturb + 2))[c(-1, -(n_disturb + 2))])
disturb[2, 2,] <- exp((rate - 0.5 * vol^2) * step + vol * sqrt(step) * quantile)
~~~

Now we are ready to perform the Bellman recursion using the fast method.

~~~
# Fast backwards induction
r_index <- matrix(c(2, 2), ncol = 2)
bellman <- FastBellman(grid, reward, control, disturb, weight, r_index)
~~~

The list **bellman** contains our approximations of the value functions, 
continuation value functions and prescribed policy at each grid point. The
value function of the option can be plotted using the following.

~~~
plot(grid[,2], rowSums(bellman$value[,,2,1] * grid), type = "l", xlab = "Stock Price", ylab = "value", main = "Option Value") 
~~~

## Example: Bermuda Put Option - Bounds

Having computed the function approximations above, we can now calculate
the bounds on the value of the option. This is performed using a pathwise
dynamic programming approach. Suppose that the current price of the underlying stock 
is **36**.

~~~
start <- c(1, 36) ## starting state
~~~

We then generate a set of sample paths for the price and disturbances for 
the nested simulation. They will be used to calculate the required martingale
increments.

~~~
set.seed(123)
# Paths
n_path <- 500
path_disturb <- array(0, dim = c(2, 2, n_dec - 1, n_path))
path_disturb[1,1,,] <- 1
rand1 <- rnorm(((n_dec - 1) * n_path) / 2)
rand1 <- c(rand1, -rand1)
path_disturb[2,2,,] <- exp((rate - 0.5 * vol^2) * step + vol * sqrt(step) * rand1)
path <- Path(start, path_disturb)

# Subsimulation
n_subsim <- 500
subsim <- array(0, dim = c(2, 2, n_subsim, n_path, n_dec - 1))
subsim[1,1,,,] <- 1
rand2 <- rnorm((n_subsim * n_path * (n_dec - 1))/2)
rand2 <- as.vector(rbind(rand2, -rand2))
subsim[2,2,,,] <- exp((rate - 0.5 * vol^2) * step + vol * sqrt(step) * rand2)
subsim_weight <- rep(1/n_subsim, n_subsim)

# Martingale increments
mart <- Martingale(bellman$value, subsim, subsim_weight, path, control)
~~~

Unlike the approximate Bellman recursion which uses subgradient
approximations of the reward functions, the diagnostics is based on
the the exact representation of the rewards, which are defined as an
ordinary function in R (which returns its values in a vectorized form
whose details are given in the manual).

~~~
# Reward function
RewardFunc <- function(state, time) {
    output <- array(data = 0, dim = c(nrow(state), 4))
    output[,4] <- exp(-0.06 * 0.02 * (time - 1)) * pmax(40 - state[,2], 0)
    return(output)
}
~~~

Having obtained the prescribed policy for our sample paths using

~~~
policy <- PathPolicy(path, control = control, Reward = RewardFunc, expected = bellman$expected, grid = grid)
~~~

the pathwise methods is then used to get the **95%** confidence interval for the
price of the option.

~~~
# 95% confidence bounds
duality <- Duality(path, control, RewardFunc, mart, policy)
GetBounds(duality, 0.05, 2)
~~~

The above gives us the interval **(4.475202, 4.480843)**. 

## Conclusion

Tighter bounds can be achived by either improving the value function approximations by using:
* a more dense grid
* a larger disturbance sampling

or by obtaining more suitable martingale increments through 
* a larger number of sample paths
* a larger number of nested simulations

The above methods are very versatile and we have employed it for the purpose of option valuation,
optimal resource extraction, partially observable Markov decision processes and optimal 
battery control.
