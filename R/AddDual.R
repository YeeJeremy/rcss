## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Additive duals using row rearrangment
################################################################################

AddDual <- function(path, subsim, weight, value, Scrap) {
    .Call('_rcss_AddDual', PACKAGE = 'rcss', path, subsim, weight, value, Scrap)
}
