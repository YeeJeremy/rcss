// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for bellman optimal function
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_BELLMAN_H_
#define INST_INCLUDE_BELLMAN_H_

#include <RcppArmadillo.h>

// Determining the optimal action and corresponding subgradient
void BellmanOptimal(const arma::mat& grid,
                    const arma::imat& control,
                    arma::cube& value,
                    const arma::cube& reward,
                    arma::cube& cont,
                    const int& dec);

// Determine the optimal action when position control non-deterministic
void BellmanOptimal2(const arma::mat& grid,
                     const arma::cube& control2,
                     arma::cube& value,
                     const arma::cube& reward,
                     arma::cube& cont,
                     const int& dec);

#endif  // INST_INCLUDE_BELLMAN_H_
