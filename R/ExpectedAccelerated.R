## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Expected value using nearest neighbours + row-rearrangement
################################################################################

ExpectedAccelerated <- function(grid, value, disturb, weight, k, Neighbour) {
    ## Making sure the input are in an acceptable format
    if (k < 2) stop("k < 2")
    d_dims <- dim(disturb)
    if (length(d_dims) != 3) stop("length(dim(disturb)) != 3")
    if (d_dims[1] != d_dims[2]) stop("dim(disturb)[1] != dim(disturb)[2]")
    if (d_dims[1] != ncol(grid)) stop("dim(disturb)[1] != ncol(grid)")
    if (length(weight) != d_dims[3]) stop("length(weight) != dim(disturb)[3]")
    ## Call the C++ function
    if (missing(Neighbour)) {
        Neighbour <- function(query, ref) {
            rflann::Neighbour(query, ref, k, "kdtree", 0, 1)$indices
        }       
    }
    .Call('rcss_ExpectedAccelerated', PACKAGE = 'rcss', grid, value, disturb,
          weight, k, Neighbour)
}
