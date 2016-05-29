## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Computing the martingale increments using nearest neighbours
################################################################################

FastMartingale2 <- function(grid, value, expected, path_disturb, path_nn,
                            control, Neighbour, path) {

    ## Making sure the input are in an acceptable format
    v_dims <- dim(value)
    if (length(v_dims) != 4) stop("length(dim(value)) != 4")
    if (ncol(grid) != v_dims[2]) stop("ncol(grid) != dim(value)[2]")
    if (nrow(grid) != v_dims[1]) stop("ncol(grid) != dim(value)[1]")
    e_dims <- dim(expected)
    if (!all(v_dims == e_dims)) stop("!all(dim(value) == dim(expected))")
    ## Making sure the input are in an acceptable format
    v_dims <- dim(value)
    if (length(v_dims) != 4) stop("length(dim(value)) != 4")
    if (ncol(grid) != v_dims[2]) stop("ncol(grid) != dim(value)[2]")
    if (nrow(grid) != v_dims[1]) stop("ncol(grid) != dim(value)[1]")
    e_dims <- dim(expected)
    if (!all(v_dims == e_dims)) stop("!all(dim(value) == dim(expected))")
    ## Call the C++ functions
    if (missing(path_nn)) {
        cat("\nComputing path_nn...")
        query <- matrix(data = path, ncol = v_dims[2])
        path_nn <- rflann::Neighbour(query, grid, 1, "kdtree", 0, 1)$indices
    }
    if (missing(Neighbour)) {
        ## Otherwise use rflann
        Func <- function(query, ref) {
            rflann::Neighbour(query, ref, 1, "kdtree", 0, 1)$indices
        }
    }
    output <- .Call('rcss_FastMartingale2', PACKAGE = 'rcss', grid, value,
                    expected, path_disturb, path_nn, Func, control)
    if (length(dim(control)) == 2) {  ## 3 dim if full control
        return(output)
    } else if (length(dim(control)) == 3) {  ## 4 dim if partial control
        n_dec <- v_dims[4]
        n_pos <- v_dims[3]
        n_path <- (dim(path_nn)[1])/(dim(value)[4])
        return(array(output, dim = c(n_dec - 1, dim(control)[2],
                                     n_path, n_pos)))
    }
}
