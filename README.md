# *rcss* - R Package

## Description

R package for approximating the value functions in Markov decision
processes (under linear state dynamics, convex reward and convex scrap
functions) using convex piecewise linear functions.  Requires the
*rflann* package (also found on my github account). Please contact me
by email (jeremyyee@outlook.com.au) or through my github account to
report any issues. An implementation in the *Julia* language is also
available on my GitHub page.

## Problem setting

We impose the following restrictions on our Markov decision process:
* a finite number of time points;
* a Markov process consisting of:
      1. a controlled Markov chain with a finite number of possible realizations (positions); 
      2. a continuous process that evolves linearly i.e. **X<sub>t+1</sub> = W<sub>t+1</sub> X<sub>t</sub>**
      where **W<sub>t+1</sub>** is a matrix with random entries;
* reward and scrap functions that are convex and Lipschitz continuous
  in the continuous process;
* a finite number of actions.

We can then approximate all the functions in the Bellman recursion
using a piece-wise linear approximation. Lower and upper bounds are
computed using a primal-dual approach.

## Example: Bermuda put option

### Value function approximation

Let us consider the valuation of a Bermuda put option with strike
price **40** and expiry date of **1** year. The underlying asset price
follows a geometric Brownian motion. We assume the option is
exercisable at 51 evenly spaced time points, including one at
beginning and one at the end of the year.

~~~
# Parameters
rate <- 0.06 ## Interest rate
step <- 0.02 ## Time step between decision epochs
vol <- 0.2 ## Volatility of stock price process
n_dec <- 51 ## Number of decision epochs
# Deterministic position control
control <- matrix(c(c(1, 1), c(2, 1)), nrow = 2, byrow = TRUE)
~~~

Define an equally spaced grid ranging from 20 to 60 and comprising
181 grid points.

~~~
# Grid
n_grid <- 241
grid <- as.matrix(cbind(rep(1, n_grid), seq(20, 60, length = n_grid)))
~~~

Introduce the reward and scrap functions in terms of sub-gradients:

~~~
## Subgrad rep of reward
strike <- 40
in_money <- grid[,2] <= strike
reward <- array(0, dim = c(n_grid, 2, 2, 2, n_dec - 1))       
reward[in_money, 1, 2, 2,] <- strike
reward[in_money, 2, 2, 2,] <- -1
for (tt in 1:(n_dec - 1)){
  reward[,,,,tt] <- exp(-rate * step * (tt - 1)) * reward[,,,,tt] 
}

## Subgrad rep of scrap
scrap <- array(data = 0, dim = c(n_grid, 2, 2))
scrap[in_money, 1, 2] <- strike
scrap[in_money, 2, 2] <- -1
scrap <- exp(-rate * step * (n_dec - 1)) * scrap
~~~

where **reward[1, ,p, a, t]** represents the intercept and **reward[,
,p, a, t]** gives the slope associated with position **p**, action
**a** and time **t**. Finally, we define the sampling of disturbances
**(W<sub>t</sub>)** which the code assumes tho be indentically
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
bellman <- FastBellman(grid, reward, scrap, control, disturb, weight, r_index)
~~~

The list **bellman** contains our approximations of the value
functions and continuation value functions at each grid point. The
value function of the option can be plotted using the following.

~~~
plot(grid[,2], rowSums(bellman$value[,,2,1] * grid), type = "l", xlab = "Stock Price", ylab = "value", main = "Option Value") 
~~~

### Primal-dual bounds

Having computed the function approximations above, we can now
calculate the bounds on the value of the option. Suppose that the
current price of the underlying stock is **36**. We simulate paths
from this starting point.

~~~
start <- c(1, 36) ## starting state
set.seed(12345)
## Paths
n_path <- 1000
start <- c(1, 36) ## starting state
path_disturb <- array(0, dim = c(2, 2, n_path, n_dec - 1))
path_disturb[1, 1,,] <- 1
rand1 <- rnorm(n_path * (n_dec - 1) / 2)
rand1 <- as.vector(rbind(rand1, -rand1))  ## anti-thetic disturbances
path_disturb[2, 2,,] <- exp((rate - 0.5 * vol^2) * step + vol * sqrt(step) * rand1)
path <- PathDisturb(start, path_disturb)
~~~

Define the exact reward and scrap functions.

~~~
## Reward function
RewardFunc <- function(state, time) {
    output <- array(data = 0, dim = c(nrow(state), 2, 2))
    output[,2, 2] <- exp(-rate * step * (time - 1)) * pmax(40 - state[,2], 0)
    return(output)
}

## Scrap function
ScrapFunc <- function(state) {
    output <- array(data = 0, dim = c(nrow(state), 2))
    output[,2] <- exp(-rate * step * (n_dec - 1)) * pmax(40 - state[,2], 0)
    return(output)
}
~~~

Now extract the prescibed policy for our paths.

~~~
policy <- FastPathPolicy(path, grid, control, RewardFunc, bellman$expected)
~~~

We then generate a set of disturbances for the nested simulation. They
will be used to calculate the required martingale increments (additive
duals).

~~~
## Subsimulation disturbances
n_subsim <- 1000
subsim <- array(0, dim = c(2, 2, n_subsim, n_path, (n_dec - 1)))
subsim[1,1,,,] <- 1
rand2 <- rnorm(n_subsim * n_path * (n_dec - 1)/ 2)
rand2 <- as.vector(rbind(rand2, -rand2))
subsim[2,2,,,] <- exp((rate - 0.5 * vol^2) * step + vol * sqrt(step) * rand2)
subsim_weight <- rep(1/n_subsim, n_subsim)

# Additive duals
mart <- FastAddDual(path, subsim, subsim_weight, grid, bellman$value, ScrapFunc)
~~~

The following duality approach is then used to get the **95%**
confidence interval for the price of the option.

~~~
# 95% confidence bounds
bounds <- AddDualBounds(path, control, RewardFunc, ScrapFunc, mart, policy)
print(GetBounds(bounds, 0.05, 2))
~~~

The above gives us the interval **(4.477726, 4.479609)**. 

## Conclusion

Tighter bounds can be achived by either improving the value function
approximations by using:
* a more dense grid
* a larger disturbance sampling

or by obtaining more suitable martingale increments through 
* a larger number of sample paths
* a larger number of nested simulations

The above methods are very versatile and we have employed it for the
purpose of option valuation, optimal resource extraction, partially
observable Markov decision processes and optimal battery control.
