## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Perform Bellman recursion
################################################################################

Bellman <- function(grid, reward, control, disturb, weight) {
    ## Making sure the input are in an acceptable format
    r_dims <- dim(reward)
    if (length(r_dims) != 5) stop("length(dim(reward)) != 5)")
    if (r_dims[1] != nrow(grid)) stop("dim(reward)[1] != nrow(grid)")
    if (r_dims[2] != ncol(grid)) stop("dim(reward)[2] != ncol(grid)")
    ##if (any(dim(control) != dim(reward)[3:4])) {
    ##    stop("any(dim(control) != dim(reward)[3:4])")
    ##}
    d_dims <- dim(disturb)
    if (length(d_dims) != 3) stop("length(dim(disturb)) != 3")
    if (d_dims[1] != d_dims[2]) stop("dim(disturb)[1] != dim(disturb)[2]")
    if (d_dims[1] != ncol(grid)) stop("dim(disturb)[1] != ncol(grid)")
    if (length(weight) != d_dims[3]) stop("length(weight) != dim(disturb)[3]")
    ## Call the C++ function
    cat('\nStarting bellman recursion...')
    output <- .Call('rcss_Bellman', PACKAGE = 'rcss', grid, reward, control,
                    disturb, weight)
    cat('Done.\n')
    ## Modify the output into an acceptable format
    n_grid <- nrow(grid)
    n_dim <- ncol(grid)
    n_position <- dim(reward)[3]
    n_dec <- dim(reward)[5]
    output$value <- array(output$value, dim = c(n_grid, n_dim,
                                            n_position, n_dec))
    output$expected <- array(output$expected, dim = c(n_grid, n_dim,
                                                  n_position, n_dec))

    return(output)   
}
