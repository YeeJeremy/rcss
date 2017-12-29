## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Additive duals using nearest neighbours
################################################################################

FastAddDual <- function(path, subsim, weight, grid, value, Scrap) {
    .Call('_rcss_FastAddDual', PACKAGE = 'rcss', path, subsim, weight, grid, value, Scrap)
}
