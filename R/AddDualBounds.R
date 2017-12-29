## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Computing the bounds
################################################################################

AddDualBounds <- function(path, control, Reward, Scrap, dual, policy) {
    .Call('_rcss_AddDualBounds', PACKAGE = 'rcss', path, control, Reward, Scrap,
          dual, policy)
}
