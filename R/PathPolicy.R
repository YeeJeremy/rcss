## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Finds the assigned policy for the paths (using row rearrangement)
################################################################################

PathPolicy <- function(path, control, Reward, expected) {
    .Call('rcss_PathPolicy', PACKAGE = 'rcss', path, control, Reward,
          expected)
}
