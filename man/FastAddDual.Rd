\name{FastAddDual}
\alias{FastAddDual}
\title{Fast additive duals}
\description{Additive duals using nearest neighbours.}
\usage{
FastAddDual(path, subsim, weight, grid, value, Scrap)
}
\arguments{
  \item{path}{3-D array representing sample paths.
    Entry [i,,j] represents the state at time j for sample path i.}
  \item{subsim}{5-D array containing the subsimulation disturbance
    matrices. Matrix [,,i,j,t] represents the disturbance used in
    subsimulation i on sample path j at time t.}
  \item{weight}{Array specifying the probability weights of the
    subsimulation disturbance matrices.}
  \item{grid}{Matrix representing the grid. The i-th row corresponds to
    i-th point of the grid. The j-th column captures the dimensions.
    The first column must equal to 1.}
  \item{value}{4-D array tangent approximation of the value function,
    where the intercept [i,1,p,t] and slope [i,-1,p,t]
    describes a tangent of the value function at grid point i for
    position p at time t.}
  \item{Scrap}{User supplied function to represent the scrap function.
    The function should take in the following argument:
    \itemize{
      \item{\eqn{n \times d} matrix representing the \eqn{n}
	\eqn{d}-dimensional states. }
    }
    The function should output the following:
    \itemize{
      \item{Matrix with dimensions \eqn{n \times p)} representing the
	scraps, where \eqn{p} is the number of positions. The \eqn{[i,
	p]}-th entry corresponds to the scrap at the \eqn{p}-th position
	for the \eqn{i}-th state.}
  }
}
}
\value{
  3-D array where entry [i,p,t] represents the martingale increment at
  time t for position p on sample path i.
}
\examples{
## Bermuda put option
grid <- as.matrix(cbind(rep(1, 81), c(seq(20, 60, length = 81))))
disturb <- array(0, dim = c(2, 2, 100))
disturb[1, 1,] <- 1
quantile <- qnorm(seq(0, 1, length = (100 + 2))[c(-1, -(100 + 2))])
disturb[2, 2,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * quantile)
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
r_index <- matrix(c(2, 2), ncol = 2)
bellman <- FastBellman(grid, reward, scrap, control, disturb, weight, r_index)
set.seed(12345)
start <- c(1, 36) ## starting state
path_disturb <- array(0, dim = c(2, 2, 100, 50))
path_disturb[1, 1,,] <- 1
rand1 <- rnorm(100 * 50 / 2)
rand1 <- as.vector(rbind(rand1, -rand1))  ## anti-thetic disturbances
path_disturb[2, 2,,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rand1)
path <- PathDisturb(start, path_disturb)
## Reward function
RewardFunc <- function(state, time) {
    output <- array(data = 0, dim = c(nrow(state), 2, 2))
    output[,2, 2] <- exp(-0.06 * 0.02 * (time - 1)) * pmax(40 - state[,2], 0)
    return(output)
}
policy <- FastPathPolicy(path, grid, control, RewardFunc, bellman$expected)
## Scrap function
ScrapFunc <- function(state) {
    output <- array(data = 0, dim = c(nrow(state), 2))
    output[,2] <- exp(-0.06 * 0.02 * 50) * pmax(40 - state[,2], 0)
    return(output)
}
## Subsimulation disturbances
subsim <- array(0, dim = c(2, 2, 100, 100, 50))
subsim[1,1,,,] <- 1
rand2 <- rnorm(100 * 100 * 50 / 2)
rand2 <- as.vector(rbind(rand2, -rand2))
subsim[2,2,,,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rand2)
subsim_weight <- rep(1 / 100, 100)
## Additive duals
mart <- FastAddDual(path, subsim, subsim_weight, grid, bellman$value, ScrapFunc)
}
\author{Jeremy Yee}

