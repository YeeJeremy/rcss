\name{FastExpected}
\alias{FastExpected}
\title{Fast Expected Value Function}
\description{Approximate the expected value function using fast methods.}
\usage{
FastExpected(grid, value, disturb, weight, r_index, Neighbour,
            smooth = 1, SmoothNeighbour)
}
\arguments{
  \item{grid}{Matrix representing the grid, whose i-th row matrix [i,]
    corresponds to i-th point of the grid. The matrix [i,1] equals to 1 while
    the vector [i,-1] represents the system state.}
  \item{value}{Matrix representing the subgradient envelope of the
    future value function, where the intercept [i,1] and
    slope matrix [i,-1] describes a subgradient at grid point i.}  
  \item{disturb}{3-dimensional array containing the disturbance
    matrices. Matrix [,,i] specifies the i-th disturbance matrix.}  
  \item{weight}{Array containing the probability weights of the
    disturbance matrices.}
  \item{r_index}{Matrix representing the positions of random entries in
    the disturbance matrix, where entry [i,1] is the row number and [i,2] gives
    the column number of the i-th random entry.} 
  \item{Neighbour}{Optional function to find the nearest neighbours. If not
    provided, the Neighbour function from the rflann package is used
    instead.}
  \item{smooth}{The number of nearest neighbours used to smooth the
    expected value functions during the Bellman recursion.}
  \item{SmoothNeighbour}{Optional function to find the nearest
    neighbours for smoothing purposes. If not provided, the Neighbour
    function from the rflann package is used instead.}  
}
\value{
  Matrix representing the subgradient envelope of the expected
  value function. Same format as the value input.
}
\examples{
## Bermuda put option
grid <- as.matrix(cbind(rep(1, 91), c(seq(10, 100, length = 91))))
disturb <- array(0, dim = c(2, 2, 10))
disturb[1,1,] <- 1
disturb[2,2,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rnorm(10))
weight <- rep(1 / 10, 10)
control <- matrix(c(c(1, 1), c(2, 1)), nrow = 2, byrow = TRUE)
reward <- array(0, dim = c(91, 2, 2, 2, 51))
reward[grid[,2] <= 40,1,2,2,] <- 40
reward[grid[,2] <= 40,2,2,2,] <- -1
r_index <- matrix(c(2, 2), ncol = 2)
bellman <- FastBellman(grid, reward, control, disturb, weight, r_index)
expected <- FastExpected(grid, bellman$value[,,2,2], disturb, weight, r_index)
}
\author{Jeremy Yee}

