## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Finding the expected value function using fast methods
################################################################################

FastExpected <- function(grid, value, disturb, weight, r_index,
                         Neighbour, smooth = 1, SmoothNeighbour) {
    ## Making sure inputs are in correct format
    if (ncol(r_index) != 2) stop("ncol(r_index) != 2")
    if (smooth < 1) stop("smooth must be >= 1")
    if (smooth >= nrow(grid)) stop("smooth must be < nrow(grid)")
    ## Call the C++ functions
    if (missing(Neighbour)) {
        Neighbour <- function(query, ref) {
            rflann::Neighbour(query, ref, 1, "kdtree", 0, 1)$indices
        }
    }
    if (missing(SmoothNeighbour)) {
        SmoothNeighbour <- function(query, ref) {
            rflann::Neighbour(query, ref, smooth, "kdtree", 0, 1)$indices
        }
    }   
    .Call('rcss_FastExpected', PACKAGE = 'rcss', grid, value, r_index,
          disturb, weight, Neighbour, smooth, SmoothNeighbour)
}
