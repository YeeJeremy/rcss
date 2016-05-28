## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## FInds the expected value function
################################################################################

Expected <- function(grid, value, disturb, weight) {
    ## Making sure the input are in an acceptable format
    d_dims <- dim(disturb)
    if (length(d_dims) != 3) stop("length(dim(disturb)) != 3")
    if (d_dims[1] != d_dims[2]) stop("dim(disturb)[1] != dim(disturb)[2]")
    if (d_dims[1] != ncol(grid)) stop("dim(disturb)[1] != ncol(grid)")
    if (length(weight) != d_dims[3]) stop("length(weight) != dim(disturb)[3]")
    ## Call the C++ function
    .Call('rcss_Expected', PACKAGE = 'rcss', grid, value, disturb, weight)
}
