\name{StochasticGrid}
\alias{StochasticGrid}
\title{Stochastic grid}
\description{Generate a grid using k-means clsutering.}
\usage{
StochasticGrid(start, disturb, n_grid, max_iter, warning)
}
\arguments{
  \item{start}{Array representing the start. The first entry must be 1
    and array [-1] represents the starting state.}
  \item{disturb}{4 dimensional array containing the path disturbances.
    Matrix [,,i,j] represents the disturbance at time i for sample path j.}
  \item{n_grid}{Number of grid points in the stochastic grid.}
  \item{max_iter}{Maximum iterations in the k-means clustering algorithm.}
  \item{warning}{Boolean indicating whether messages from the k-means
    clustering algorithm are to be displayed}
}
\value{
  Matrix representing the stochastic matrix. Each row represents a
  particular grid point. The first column contains only 1s. 
}
\examples{
## Generate a stochastic matrix using an AR(2) process
start <- c(1, 0, 0)
n_dim <- length(start)
n_path <- 10
psi1 <- 0.3
psi2 <- 0.65
n_dec <- 21
path_disturb <- array(data = matrix(c(1, 0,    0,
                                      0, 0,    1,
                                      0, psi2, psi1), ncol = 3, byrow = TRUE),
                      dim = c(n_dim, n_dim, (n_dec - 1), n_path))
path_disturb[3,1,,] <- runif((n_dec - 1) * n_path, -1, 1)
n_grid <- 10
grid <- StochasticGrid(start, path_disturb, n_grid, 10, TRUE)
}
\author{Jeremy Yee}

