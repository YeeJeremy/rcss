## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Computing the bounds
################################################################################

AddDualBounds <- function(path, control, Reward, Scrap, add_dual, policy) {
    .Call('rcss_AddDualBounds', PACKAGE = 'rcss', path, control, Reward, Scrap,
          add_dual, policy)
}
