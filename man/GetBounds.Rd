\name{GetBounds}
\alias{GetBounds}
\title{Confidence Bounds}
\description{Confidence bounds for the value.}
\usage{
GetBounds(duality, alpha, position)
}
\arguments{
  \item{duality}{Object returned by the Duality function.}
  \item{alpha}{Specifies the (1-alpha) confidence bounds.}
  \item{position}{Natural number indicating the starting position.}
}
\value{
  Array representing the (1-alpha) confidence bounds for the value
  of the specified position.
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
subsim_disturb <- array(0, dim = c(2, 2, 20, 100, 50))
subsim_disturb[1,1,,,] <- 1
subsim_disturb[2,2,,,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rnorm(100000))
subsim_weight <- rep(1 / 20, 20)
mart <- FastMartingale(bellman$value, path, path_nn, subsim_disturb,
subsim_weight, grid, control = control)
RewardFunc <- function(state, time) {
    output <- array(data = 0, dim = c(nrow(state), 4))
    output[,4] <- pmax(40 - state[,2], 0)
    return(output)
}
path_action <- PathPolicy(path, path_nn, control, RewardFunc, bellman$expected)
duality <- Duality(path, control, RewardFunc, mart, path_action)
bounds <- GetBounds(duality, 0.05, 2)
}
\author{Jeremy Yee}

