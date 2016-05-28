## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Computing the primal and dual values
################################################################################

Duality <- function(path, control, Reward, mart, path_action) {
    ## Making sure the input are in an acceptable format
    p_dims = dim(path)
    if (length(p_dims) != 3) stop("length(dim(path)) != 3")
    ##m_dims = dim(mart)
    ##if (length(m_dims) != 3) stop("length(dim(mart)) != 3")
    ##if (m_dims[1] != p_dims[1] - 1) stop("dim(mart)[1] != dim(path)[1] - 1")
    ##if (m_dims[2] != nrow(control)) stop("dim(mart)[2] != dim(control)[1]")
    ##if (m_dims[3] != p_dims[2]) stop("dim(mart)[3] != dim(path)[2]")
    ## Calling C++ functions
    .Call('rcss_Duality', PACKAGE = 'rcss', path, control, Reward, mart,
          path_action)
}
