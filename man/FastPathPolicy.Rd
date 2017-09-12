\name{FastPathPolicy}
\alias{FastPathPolicy}
\title{Fast prescribed policy}
\description{Policy precribed to provided sample paths using nearest neighbours}
\usage{
FastPathPolicy(path, grid, control, Reward, expected)
}
\arguments{
  \item{path}{3-D array representing sample paths.
    Entry [i,,j] represents the state at time j for sample path i.}
  \item{grid}{Matrix representing the grid. The i-th row corresponds to
    i-th point of the grid. The j-th column captures the dimensions.
    The first column must equal to 1.}
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
  \item{Reward}{User supplied function to represent the reward function.
    The function should take in the following arguments, in this order:
    \itemize{
      \item{\eqn{n \times d} matrix representing the \eqn{n}
	\eqn{d}-dimensional states. }
      \item{A natural number representing the decison epoch.}
    }
    The function should output the following:
    \itemize{
      \item{3-D array with dimensions \eqn{n \times (a \times p)}
	representing the rewards, where \eqn{p} is the number of
	positions and \eqn{a} is the number of actions in the
	problem. The \eqn{[i, a, p]}-th entry corresponds to the reward
	from applying the \eqn{a}-th action to the \eqn{p}-th position
	for the \eqn{i}-th state.}  }
  }
  \item{expected}{4-D array representing the tangent approximation of the
    expected value function, where the intercept [i,1,p,t] and slope
    [i,-1,p,t] describes a tangent at grid point i for position p
    at time t.}
}
\value{
  3-D array representing the prescribed policy for the sample
  paths. Entry [i,p,t] gives the prescribed action at time t for
  position p on sample path t.
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
}
\author{Jeremy Yee}



