## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Generate a stochastic grid using K-Means clustering
################################################################################

StochasticGrid <- function(path, n_grid, max_iter, warning = FALSE) {
    .Call('_rcss_StochasticGrid', PACKAGE = 'rcss', path, n_grid, max_iter, warning)
}
