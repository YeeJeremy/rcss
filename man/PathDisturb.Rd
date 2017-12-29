\name{PathDisturb}
\alias{PathDisturb}
\title{Path Simulation}
\description{Simulate sample paths using disturbances.}
\usage{
PathDisturb(start, disturb)
}
\arguments{
  \item{start}{Array representing the start. The first entry must be 1
    and array [-1] represents the starting state.}
  \item{disturb}{4-dimensional array containing the path disturbances.
    Matrix [,,i,j] represents the disturbance at time j for sample path i.}
}
\value{
  3-dimensional array representing the generated paths. Array [i,,j]
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
                      dim = c(n_dim, n_dim, n_path, (n_dec - 1)))
path_disturb[3,1,,] <- runif(n_path * (n_dec - 1), -1, 1)
path <- PathDisturb(start, path_disturb)
}
\author{Jeremy Yee}

