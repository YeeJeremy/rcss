## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Perform fast Bellman recursion
################################################################################

FastBellman <- function(grid, reward, control, disturb, weight, r_index,
                        Neighbour, smooth = 1, SmoothNeighbour) {
    ## Making sure the inputs are in correct format
    r_dims <- dim(reward)
    if (length(r_dims) != 5) stop("length(dim(reward)) != 5)")
    if (r_dims[1] != nrow(grid)) stop("dim(reward)[1] != nrow(grid)")
    if (r_dims[2] != ncol(grid)) stop("dim(reward)[2] != ncol(grid)")
    ##if (any(dim(control) != dim(reward)[3:4])) {
    ##    stop("any(dim(control) != dim(reward)[3:4])")
    ##}
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
    output <- .Call('rcss_FastBellman', PACKAGE = 'rcss', grid, reward,
                    control, r_index, disturb, weight, Neighbour, smooth,
                    SmoothNeighbour)
    ## Put output into correct format
    n_grid <- nrow(grid)
    n_dim <- ncol(grid)
    n_position <- dim(reward)[3]
    n_dec <- dim(reward)[5]
    output$value <- array(data = output$value, dim = c(n_grid, n_dim,
                                                   n_position, n_dec))
    output$expected <- array(data = output$expected, dim = c(n_grid, n_dim,
                                                         n_position, n_dec))   
    cat("Done.\n")
    return(output)
}
