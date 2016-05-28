\name{FastMartingale2}
\alias{FastMartingale2}
\title{Fast Approximate Martingales -- Experimental}
\description{Experimental function. Will be studied in a subsequent paper.}
\usage{
FastMartingale2(grid, value, expected, path_disturb, path_nn, control,
                Neighbour, path)
}
\arguments{
  \item{grid}{Matrix representing the grid, whose i-th row matrix [i,]
    corresponds to i-th point of the grid. The matrix [i,1] equals to 1 while
    the vector [i,-1] represents the system state.}
  \item{value}{4-dimensional array representing the subgradient
    envelope of the value function, where the intercept [i,1,p,t] and
    slope matrix [i,-1,p,t] describes a subgradient of the value function
    at grid point i for position p at time t.}
  \item{expected}{4-dimensional array representing the expected
    value functions. Same format as the value array.} 
  \item{path_disturb}{4-dimensional array containing the path disturbances.
    Matrix [,,i,j] represents the disturbance at time i for sample path j.}
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
  \item{Neighbour}{Optional function to calculate the nearest
    neighbours. If not provided, the Neighbour function from the
    rflann package is used instead.}
  \item{path}{3-dimensional array representing the generated paths.
    Array [i,j,] represents the state at time i for sample path j.}
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
control_mat <- matrix(c(c(1, 1), c(2, 1)), nrow = 2, byrow = TRUE)
reward <- array(0, dim = c(91, 2, 2, 2, 51))
reward[grid[,2] <= 4,1,2,2,] <- 40
reward[grid[,2] <= 4,2,2,2,] <- -1
r_index <- matrix(c(2, 2), ncol = 2)
bellman <- FastBellman(grid, reward, control_mat, disturb, disturb_weight, r_index)
path_disturb <- array(0, dim = c(2, 2, 50, 100))
path_disturb[1,1,,] <- 1
path_disturb[2,2,,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rnorm(5000))
start <- c(1, 36)
path <- Path(start, path_disturb)
path_nn <- Neighbour(matrix(path, ncol = 2), grid, 1, "kdtree", 0, 1)$indices
mart <- FastMartingale2(grid, bellman$value, bellman$expected,
                        path_disturb, path_nn, control_mat, path = path)
}
\author{Jeremy Yee}

