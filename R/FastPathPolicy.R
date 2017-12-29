## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Finds the assigned policy for the paths using nearest neighbours
################################################################################

FastPathPolicy <- function(path, grid, control, Reward, expected) {
    .Call('_rcss_FastPathPolicy', PACKAGE = 'rcss', path, grid, control, Reward,
          expected)
}
