## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Perform Bellman recursion
################################################################################

Bellman <- function(grid, reward, scrap, control, disturb, weight) {
    output <- .Call('rcss_Bellman', PACKAGE = 'rcss', grid, reward, scrap,
                    control, disturb, weight)
    ## Put output into correct format
    n_grid <- nrow(grid)
    n_dim <- ncol(grid)
    n_position <- dim(reward)[3]
    n_dec <- dim(reward)[5] + 1
    dimens <- c(n_grid, n_dim, n_position, n_dec)
    dimens1 <- c(n_grid, n_dim, n_position, n_dec - 1)
    output$value <- array(output$value, dim = dimens)
    output$expected <- array(output$expected, dim = dimens1)
    cat("Done.\n")
    return(output)   
}
