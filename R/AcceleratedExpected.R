## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Expected value using nearest neighbours
################################################################################

AcceleratedExpected <- function(grid, value, disturb, weight, k = 1) {
    .Call('_rcss_AcceleratedExpected', PACKAGE = 'rcss', grid, value, disturb,
          weight, k)
}
