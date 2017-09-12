## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Expected value using nearest neighbours
################################################################################

AcceleratedExpected <- function(grid, value, disturb, weight, k = 1) {
    .Call('rcss_AcceleratedExpected', PACKAGE = 'rcss', grid, value, disturb,
          weight, k)
}
