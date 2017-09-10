## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Testing the prescribed policy and return position evolution
################################################################################

FullTestPolicy <- function(position, path, control, Reward, Scrap, path_action) {
    .Call('rcss_FullTestPolicy', PACKAGE = 'rcss', position, path,
          control, Reward, Scrap, path_action)
}
