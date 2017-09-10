// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Row rearrangement operator
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_SLOW_H_
#define INST_INCLUDE_SLOW_H_

#include <RcppArmadillo.h>

// Finds the maximising subgradient
arma::mat Optimal(const arma::mat& grid,
                  const arma::mat& subgradient);

#endif  // INST_INCLUDE_SLOW_H_
