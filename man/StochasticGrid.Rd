\name{StochasticGrid}
\alias{StochasticGrid}
\title{Stochastic grid}
\description{Generate a grid using k-means clustering.}
\usage{
StochasticGrid(path, n_grid, max_iter, warning)
}
\arguments{
  \item{path}{3-D array representing sample paths.  Entry [i,,j]
    represents the state at time j for sample path i.}
  \item{n_grid}{Number of grid points in the stochastic grid.}
  \item{max_iter}{Maximum iterations in the k-means clustering
  algorithm.}
  \item{warning}{Boolean indicating whether messages from the k-means
    clustering algorithm are to be displayed}
}
\value{
  Matrix representing the stochastic grid. Each row represents a
  particular grid point. The first column contains only 1. 
}
\examples{
## Generate paths
start <- c(1, 36)
path_disturb <- array(0, dim = c(2, 2, 100, 50))
path_disturb[1, 1,,] <- 1
rand1 <- rnorm((50 * 100) / 2)
rand1 <- as.vector(rbind(rand1, -rand1))
path_disturb[2, 2,,] <- exp((0.06 - 0.5 * 0.2^2) * 0.02 + 0.2 * sqrt(0.02) * rand1)
path <- PathDisturb(start, path_disturb)
sgrid <- StochasticGrid(path, 81, 10)
}
\author{Jeremy Yee}

