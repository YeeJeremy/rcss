\name{Path}
\alias{Path}
\title{Path Simulation}
\description{Simulate sample paths.}
\usage{
Path(start, disturb)
}
\arguments{
  \item{start}{Array representing the start. The first entry must be 1
    and array [-1] represents the starting state.}
  \item{disturb}{4-dimensional array containing the path disturbances.
    Matrix [,,i,j] represents the disturbance at time i for sample path j.}
}
\value{
  3-dimensional array representing the generated paths. Array [i,j,]
  represents the state at time i for sample path j.
}
\examples{
## Simulating AR(2) process
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
path <- Path(start, path_disturb)
}
\author{Jeremy Yee}

