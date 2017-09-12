## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Row rearrangement operator
################################################################################

Optimal <- function(grid, tangent) {
    .Call('rcss_Optimal', PACKAGE = 'rcss', grid, tangent)
}
