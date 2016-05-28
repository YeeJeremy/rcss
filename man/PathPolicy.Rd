\name{PathPolicy}
\alias{PathPolicy}
\title{Prescribed Policy}
\description{Policies precribed to selected sample paths.}
\usage{
PathPolicy(path, path_nn, control, Reward, expected, grid)
}
\arguments{
  \item{path}{3-dimensional array representing the generated paths.
    Array [i,j,] represents the state at time i for sample path j.}
  \item{path_nn}{Array containing the nearest neighbours, where the
    (dim(path)[1]*(i-1) + j)-th entry corresponds to the nearest
    neighbour for sample path i at time j. Optional if grid and path are
    supplied.}
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
  \item{Reward}{User supplied function to represent the reward function.
    The function should take in the following arguments, in this order:
    \itemize{
      \item{\eqn{n \times d} matrix representing the \eqn{n}
	\eqn{d}-dimensional states. }
      \item{A natural number representing the decison epoch.}
    }
    The function should output the following:
    \itemize{
      \item{\eqn{n \times (p \times a)} matrix representing the
	rewards, where \eqn{p} is the number of positions and \eqn{a} is
	the number of actions in the problem. The \eqn{[i, a \times (j
	  - 1) + k]}-th entry corresponds to the reward
	from applying the \eqn{k}-th action to the \eqn{j}-th position
	for the \eqn{i}-th state.}
    }
  }

  \item{expected}{4-dimensional array representing the subgradient
    envelope of the expected value function, where the intercept [i,1,p,t] and
    slope matrix [i,-1,p,t] describes a subgradient at grid point
    i for position p at time t.}
  \item{grid}{Matrix representing the grid, whose i-th row matrix [i,]
    corresponds to i-th point of the grid. The matrix [i,1] equals to 1 while
    the vector [i,-1] represents the system state.}
}
\value{
  3-dimensional array representing the prescribed
  policy for the sample paths. Entry [i,j,k] gives the prescribed
  action at time i for position j on sample path k.
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
path_nn <- Neighbour(matrix(path, ncol = 2), grid, 1, "kdtree", 0, 1)$indices
RewardFunc <- function(state, time) {
    output <- array(data = 0, dim = c(nrow(state), 4))
    output[,4] <- exp(-0.06 * 0.02 * (time - 1)) * pmax(40 - state[,2], 0)
    return(output)
}
path_action <- PathPolicy(path, path_nn, control, RewardFunc, bellman$expected)
}
\author{Jeremy Yee}



