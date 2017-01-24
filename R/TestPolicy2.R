## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Testing the prescribed policy and return position evolution
################################################################################

TestPolicy2 <- function(position, path, control, Reward, path_action) {
    ## Checking inputs are in correct format
    p_dims <- dim(path)
    if (length(p_dims) != 3) stop("length(dim(path)) != 3")
    a_dims <- dim(path_action)
    if (length(a_dims) != 3) stop("length(dim(path_action)) != 3")  
    ## Call the C++ function
    return(.Call('rcss_TestPolicy2', PACKAGE = 'rcss', position, path,
                 control, Reward, path_action))
}
