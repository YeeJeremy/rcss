// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for fast methods
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_FAST_H_
#define INST_INCLUDE_FAST_H_

#include <RcppArmadillo.h>
#include <rflann.h>

// Block diagonal matrix
arma::mat BlockDiag(const arma::mat& input, std::size_t n_repeat);

// Smoothing by using nearest neighbours
arma::mat Smooth(const arma::mat& grid,
                 const arma::mat& subgradient,
                 const arma::umat& smooth_neighbour);

// Generate the conditional expectation matrices
void ExpectMat(arma::mat& constant,
               arma::cube& perm,
               const arma::mat& grid,
               const arma::umat& r_index,
               const arma::cube& disturb,
               const arma::vec& weight); 

#endif  // INST_INCLUDE_FAST_H_
