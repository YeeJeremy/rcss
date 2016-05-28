\name{Bellman}
\alias{Bellman}
\title{Bellman Recursion}
\description{Approximate the value functions using the Bellman recursion.}
\usage{
Bellman(grid, reward, control, disturb, weight)
}
\arguments{
  \item{grid}{Matrix representing the grid, whose i-th row matrix [i,]
    corresponds to i-th point of the grid. The matrix [i,1] equals to 1 while
    the vector [i,-1] represents the system state.}
  \item{reward}{5-dimensional array representing the subgradient envelope
    of the reward function. The matrix [i,,a,p,t] captures the subgradient at
    grid point i for action a taken in position p at time t, with the intercept
    given by [i,1,a,p,t] and slope by [i,-1,a,p,t].}
  \item{control}{Array representing the transition probabilities of the
    controlled Markov chain. Two possible inputs:
    \itemize{
      \item{Matrix of dimension n_pos \eqn{\times}
        n_action, where entry [i,j] describes the next
        position after selecting action j at position i.}
      \item{3-dimensional array with dimensions n_pos
	\eqn{\times} n_action \eqn{\times} n_pos, where entry [i,j,k] is
	the probability of moving to position k after applying action j
	to position i.}
    }
  }
  \item{disturb}{3-dimensional array containing the disturbance
    matrices. Matrix [,,i] specifies the i-th disturbance matrix.}
  \item{weight}{Array containing the probability weights of the
    disturbance matrices.}
}
\value{
  \item{value}{4-dimensional array representing the subgradient
    envelope of the value function, where the intercept [i,1,p,t] and
    slope matrix [i,-1,p,t] describes a subgradient of the value function
    at grid point i for position p at time t.}
  \item{expected}{4-dimensional array representing the expected
    value functions. Same format as the value array.} 
  \item{action}{3-dimensional array representing the prescribed policy,
    where entry [i,j,k] is the decision rule at grid point i for
    position j at time k.}
}
\examples{
## Bermuda put option
grid <- as.matrix(cbind(rep(1, 91), c(seq(10, 100, length = 91))))
disturb <- array(0, dim = c(2, 2, 10))
disturb[1, 1,] <- 1
disturb[2, 2,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rnorm(10))
weight <- rep(1 / 10, 10)
control <- matrix(c(c(1, 2),c(1, 1)), nrow = 2)
reward <- array(data = 0, dim = c(91, 2, 2, 2, 51))
reward[grid[, 2] <= 40, 1, 2, 2,] <- 40
reward[grid[, 2] <= 40, 2, 2, 2,] <- -1
bellman <- Bellman(grid, reward, control, disturb, weight)
}
\author{Jeremy Yee}

