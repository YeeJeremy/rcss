## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Generate paths using the user supplied disturbance matrices
################################################################################

Path <- function(start, disturb) {
    ## Checking whether the arguments are in correct format
    d_dims <- dim(disturb)
    if (length(start) != d_dims[1]) stop("length(start) != dim(disturb)[1]")
    if (length(d_dims) != 4) stop("length(d_dims) != 4")    
    if (d_dims[1] != d_dims[2]) stop("dim(disturb)[1] != dim(disturb)[2]")
    ## Call the C++ function
    return(.Call('rcss_Path', PACKAGE = 'rcss', start, disturb))
}
