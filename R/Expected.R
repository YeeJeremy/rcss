## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Expected value function using row rearrangement
################################################################################

Expected <- function(grid, value, disturb, weight) {
    .Call('rcss_Expected', PACKAGE = 'rcss', grid, value, disturb, weight)
}
