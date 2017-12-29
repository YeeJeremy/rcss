## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Generate paths using path disturbances
################################################################################

PathDisturb <- function(start, disturb) {
    .Call('_rcss_PathDisturb', PACKAGE = 'rcss', start, disturb)
}
