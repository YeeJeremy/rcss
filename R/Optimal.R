## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Row rearrangement operator
################################################################################

Optimal <- function(grid, tangent) {
    .Call('_rcss_Optimal', PACKAGE = 'rcss', grid, tangent)
}
