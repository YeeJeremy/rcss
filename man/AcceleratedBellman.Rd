\name{AcceleratedBellman}
\alias{AcceleratedBellman}
\title{Bellman recursion accelerated with k nearest neighbours}
\description{Approximate the value functions using k nearest neighbours.}
\usage{
AcceleratedBellman(grid, reward, scrap, control, disturb, weight, k = 1)
}
\arguments{
  \item{grid}{Matrix representing the grid. The i-th row corresponds to
    i-th point of the grid. The j-th column captures the dimensions.
    The first column must equal to 1.}
  \item{reward}{5-D array representing the tangent approximation of
    the reward. Entry [i,,a,p,t] captures the tangent at grid point
    i for action a taken in position p at time t. The intercept is given
    by [i,1,a,p,t] and slope by [i,-1,a,p,t].}
  \item{scrap}{3-D array representing the tangent approximation of the
    scrap. Entry [i,,p] captures the tangent at grid point i for
    position p. The intercept is given by [i,1,p] and slope by
    [i,-1,p].}
  \item{control}{Array representing the transition probabilities of the
    controlled Markov chain. Two possible inputs:
    \itemize{
      \item{Matrix of dimension n_pos \eqn{\times} n_action, where entry
        [i,j] describes the next position after selecting action j at
        position i.}
      \item{3-D array with dimensions n_pos \eqn{\times} n_action
	\eqn{\times} n_pos, where entry [i,j,k] is the probability of
	moving to position k after applying action j to position i.}
    }
  }
  \item{disturb}{3-D array containing the disturbance matrices. Matrix
    [,,i] specifies the i-th disturbance matrix.}
  \item{weight}{Array containing the probability weights of the
    disturbance matrices.}
  \item{k}{Number of nearest neighbours used for each grid point.}
}
\value{
  \item{value}{4-D array tangent approximation of the value function,
    where the intercept [i,1,p,t] and slope [i,-1,p,t] describes
    a tangent of the value function at grid point i for position p
    at time t.}
  \item{expected}{4-D array representing the expected value functions.}
}
\examples{
## Bermuda put option
grid <- as.matrix(cbind(rep(1, 81), c(seq(20, 60, length = 81))))
disturb <- array(0, dim = c(2, 2, 100))
disturb[1, 1,] <- 1
quantile <- qnorm(seq(0, 1, length = (100 + 2))[c(-1, -(100 + 2))])
disturb[2, 2,] <- exp((0.06 -0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * quantile)
weight <- rep(1 / 100, 100)
control <- matrix(c(c(1, 2),c(1, 1)), nrow = 2)
reward <- array(data = 0, dim = c(81, 2, 2, 2, 50))
in_money <- grid[, 2] <= 40
reward[in_money, 1, 2, 2,] <- 40
reward[in_money, 2, 2, 2,] <- -1
for (tt in 1:50){
  reward[,,2,2,tt] <- exp(-0.06 * 0.02 * (tt - 1)) * reward[,,2,2,tt] 
}
scrap <- array(data = 0, dim = c(81, 2, 2))
scrap[in_money, 1, 2] <- 40
scrap[in_money, 2, 2] <- -1
scrap[,,2] <- exp(-0.06 * 0.02 * 50) * scrap[,,2]
bellman <- AcceleratedBellman(grid, reward, scrap, control, disturb, weight)
}
\author{Jeremy Yee}

