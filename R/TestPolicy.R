## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Testing the prescribed policy
################################################################################

TestPolicy <- function(position, path, control, Reward, Scrap, policy) {
    .Call('rcss_TestPolicy', PACKAGE = 'rcss', position, path,
          control, Reward, Scrap, policy)
}
