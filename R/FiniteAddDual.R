## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Additive duals for finite distribution case
################################################################################

FiniteAddDual <- function(path, disturb, grid, value, expected, build = "fast", k = 1) {
    .Call('rcss_FiniteAddDual', PACKAGE = 'rcss', path, disturb, grid, value,
          expected, build, k)
}
