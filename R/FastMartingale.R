## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Computing the martingale increments using nearest neighbours
################################################################################

FastMartingale <- function(value, path, path_nn, disturb, weight, grid,
                           Neighbour, control) {
    ## Making sure the input are in an acceptable format
    v_dims = dim(value)
    if (length(v_dims) != 4) stop("length(dim(value)) != 4")
    path_dims = dim(path)
    if (length(path_dims) != 3) stop("length(dim(path)) != 3")
    if (path_dims[1] != v_dims[4]) stop("dim(path)[1] != dim(value)[4]")
    if (path_dims[3] != v_dims[2]) stop("dim(path)[3] != dim(value)[2]")    
    if (missing(disturb) != missing(weight)) {
        stop("missing(disturb) != missing(weight)")
    }  
    d_dims = dim(disturb)
    if (length(d_dims) != 5) stop("length(dim(disturb)) != 5")
    if (v_dims[2] != d_dims[1]) stop("dim(disturb)[1] != dim(value)[2]")
    if (d_dims[1] != d_dims[2]) stop("dim(disturb)[1] != dim(disturb)[2]")
    if (length(weight) != d_dims[3]) stop("length(weight) != dim(disturb)[3]")
    if (d_dims[4] != path_dims[2]) stop("dim(disturb)[4] != dim(path)[2]")
    if (d_dims[5] != (path_dims[1] - 1)) stop("dim(disturb)[5] != dim(path)[1]")
    if (ncol(grid) != v_dims[2]) stop("ncol(grid) != dim(value)[2]")
    if (path_dims[3] != d_dims[1]) stop("dim(path)[3] != dim(disturb)[1]")
    if (ncol(grid) != v_dims[2]) stop("ncol(grid) != dim(value)[2]")
    ## Call the C++ functions
    if (missing(path_nn)) {
        cat("\nComputing path_neighbour...")
        query <- matrix(data = path, ncol = v_dims[2])
        path_nn <- rflann::Neighbour(query, grid, 1, "kdtree", 0, 1)$indices
    }
    if (!missing(Neighbour)) {
        ## If user provides a function
        output <- .Call('rcss_FastMartingale', PACKAGE = 'rcss', value, disturb,
                        weight, path, path_nn, Neighbour, grid, control)
    } else {
        ## Otherwise use rflann
        Func <- function(query, ref) {
            rflann::Neighbour(query, ref, 1, "kdtree", 0, 1)$indices
        }
        output <- .Call('rcss_FastMartingale', PACKAGE = 'rcss', value, disturb,
                        weight, path, path_nn, Func, grid, control)      
    }
    if (length(dim(control)) == 2) {  ## 3 dim if full control
        return(output)
    } else if (length(dim(control)) == 3) {  ## 4 dim if partial control
        n_dec <- v_dims[4]
        n_pos <- v_dims[3]
        n_path <- path_dims[2]
        return(array(output, dim = c(n_dec - 1, dim(control)[2],
                                     n_path, n_pos)))
    }
}
