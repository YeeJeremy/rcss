## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Bellman recursion using conditional expectation matrices
################################################################################

FastBellman <- function(grid, reward, scrap, control, disturb, weight, r_index,
                        smooth = 1) {
    output <- .Call('rcss_FastBellman', PACKAGE = 'rcss', grid, reward, scrap,
                    control, r_index, disturb, weight, smooth)
    ## Put output into correct format
    n_grid <- nrow(grid)
    n_dim <- ncol(grid)
    n_position <- dim(reward)[4]
    n_dec <- dim(reward)[5] + 1
    dimens <- c(n_grid, n_dim, n_position, n_dec)
    dimens1 <- c(n_grid, n_dim, n_position, n_dec - 1)
    output$value <- array(output$value, dim = dimens)
    output$expected <- array(output$expected, dim = dimens1)
    cat("Done.\n")
    return(output)   
}
