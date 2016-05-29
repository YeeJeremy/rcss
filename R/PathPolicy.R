## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Finds the policy for the paths specified
################################################################################

PathPolicy <- function(path, path_nn, control, Reward, expected, grid) {
    ## Making sure the input are in an acceptable format
    p_dims = dim(path)
    if (length(p_dims) != 3) stop("length(dim(path)) != 3")
    e_dims = dim(expected)
    if (length(e_dims) != 4) stop("length(dim(expected)) != 4")
    if (e_dims[2] != p_dims[3]) stop("dim(expected)[2] != dim(path)[3]")
    if (e_dims[3] != nrow(control)) stop("dim(expected)[3] != dim(control)[1]")
    if (e_dims[4] != p_dims[1]) stop("dim(expected)[4] != dim(path)[1]")
    ## Calling C++ functions
    if (missing(path_nn)) {
        if (missing(grid)) stop("grid is missing")
        query <- matrix(data = path, ncol = p_dims[3])
        path_nn <- rflann::Neighbour(query, grid, 1, "kdtree", 0, 1)$indices
    }
    .Call('rcss_PathPolicy', PACKAGE = 'rcss', path, path_nn, control, Reward,
          expected)
}
