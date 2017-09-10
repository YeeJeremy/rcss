## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Additive duals using row rearrangment
################################################################################

AddDual <- function(path, subsim, weight, value, Scrap) {
    .Call('rcss_AddDual', PACKAGE = 'rcss', path, subsim, weight, value, Scrap)
}
