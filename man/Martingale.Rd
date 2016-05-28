\name{Martingale}
\alias{Martingale}
\title{Martingale Increments}
\description{Compute the martingale increments.}
\usage{
Martingale(value, disturb, weight, path, control)
}
\arguments{
  \item{value}{4-dimensional array representing the subgradient
    envelope of the value function, where the intercept [i,1,p,t] and
    slope matrix [i,-1,p,t] describes a subgradient of the value function
    at grid point i for position p at time t.}  
  \item{disturb}{5-dimensional array containing the disturbance
    matrices. Matrix [,,i,j,k] represents the disturbance used in
    sub-simulation i on sample path j at time k.}
  \item{weight}{Array specifying the probability weights of the
    disturbance matrices.}
  \item{path}{3-dimensional array representing the generated paths.
    Array [i,j,] represents the state at time i for sample path j.}
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
}
\value{
  Two possible outputs:
  \itemize{
    \item{Full control: 3-dimensional array, where entry [i,j,k]
      represents the martingale increment at time i for position j
      on sample path k.}
    \item{Partial control: 4-dimensional array, where entry [i,j,k,l]
      represents the martingale increment at time i after applying
      action j on sample path k to position l.}
  }
}
\examples{
## Bermuda put option
grid <- as.matrix(cbind(rep(1, 91), c(seq(10, 100, length = 91))))
disturb <- array(0, dim = c(2, 2, 10))
disturb[1,1,] <- 1
disturb[2,2,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rnorm(10))
disturb_weight <- rep(1 / 10, 10)
control <- matrix(c(c(1, 1), c(2, 1)), nrow = 2, byrow = TRUE)
reward <- array(0, dim = c(91, 2, 2, 2, 51))
reward[grid[,2] <= 4,1,2,2,] <- 40
reward[grid[,2] <= 4,2,2,2,] <- -1
r_index <- matrix(c(2, 2), ncol = 2)
bellman <- FastBellman(grid, reward, control, disturb, disturb_weight, r_index)
path_disturb <- array(0, dim = c(2, 2, 50, 100))
path_disturb[1,1,,] <- 1
path_disturb[2,2,,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rnorm(5000))
start <- c(1, 36)
path <- Path(start, path_disturb)
disturb <- array(0, dim = c(2, 2, 20, 100, 50))
disturb[1,1,,,] <- 1
disturb[2,2,,,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rnorm(100000))
weight <- rep(1 / 20, 20)
mart <- Martingale(bellman$value, disturb, weight, path, control)
}
\author{Jeremy Yee}

