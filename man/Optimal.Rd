\name{Optimal}
\alias{Optimal}
\title{Optimal}
\description{Find the maximising tangent at each grid point.}
\usage{
Optimal(grid, tangent)
}
\arguments{
  \item{grid}{Matrix representing the grid. The i-th row corresponds to
    i-th point of the grid. The j-th column captures the dimensions.
    The first column must equal to 1.}
  \item{tangent}{Matrix representing the collection of tangents, where
    the intercept [i,1] and slope [i,-1] describes a tangent at grid
    point i.}
}
\value{
  Matrix representing the maximum of the tangents at each grid point,
  where the intercept [i,1] and slope [i,-1] describes the maximising
  tangent at grid point i.
}
\examples{
grid <- as.matrix(cbind(rep(1, 81), c(seq(20, 60, length = 91))))
tangent <- matrix(rnorm(81 * 2), ncol = 2)
Optimal(grid, tangent)
}
\author{Jeremy Yee}

