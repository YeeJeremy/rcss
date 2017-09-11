// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for accclerated methods
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_ACCELERATED_H_
#define INST_INCLUDE_ACCELERATED_H_

#include <RcppArmadillo.h>

// Finds the maximising subgradient using nearest neighbours + row-rearrange
arma::mat OptimalNeighbour(const arma::mat& grid,
                           const arma::mat& subgradient,
                           const arma::umat& neighbour);


#endif  // INST_INCLUDE_ACCELERATED_H_
