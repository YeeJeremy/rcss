## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Expected value function using conditional expectation matrices
################################################################################

FastExpected <- function(grid, value, disturb, weight, r_index, smooth = 1) {
    .Call('rcss_FastExpected', PACKAGE = 'rcss', grid, value, r_index,
          disturb, weight,smooth)
}
