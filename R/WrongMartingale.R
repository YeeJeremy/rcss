## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Incorrect martingale increments for finite distribtuion
################################################################################

WrongMartingale <- function(value, expected, path, control) {
    ## Making sure the input are in an acceptable format
    v_dims = dim(value)
    if (length(v_dims) != 4) stop("length(dim(value)) != 4")
    e_dims = dim(expected)
    if (length(e_dims) != 4) stop("length(dim(expected)) != 4")
    path_dims = dim(path)
    if (length(path_dims) != 3) stop("length(dim(path)) != 3")
    if (path_dims[1] != v_dims[4]) stop("dim(path)[1] != dim(value)[4]")
    if (path_dims[3] != v_dims[2]) stop("dim(path)[3] != dim(value)[2]")    
    ## Call the C++ function
    output <- .Call('rcss_WrongMartingale', PACKAGE = 'rcss', value, expected,
                    path, control)
    if (length(dim(control)) == 2) {  ## 3 dim if full control
        return(output)
    } else if (length(dim(control)) == 3) {  ## 4 dim if partial control
        n_dec <- v_dims[4]
        n_pos <- v_dims[3]
        n_path <- dim(path)[2]
        return(array(output, dim = c(n_dec - 1, dim(control)[2],
                                     n_path, n_pos)))
    }
}
