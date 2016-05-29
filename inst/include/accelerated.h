// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for accclerated methods
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_ACCELERATED_H_
#define INST_INCLUDE_ACCELERATED_H_

#include <RcppArmadillo.h>

// Finds the maximising subgradient using nearest neighbours + row-rearrange
arma::mat OptimalNeighbour(const arma::mat& grid,
                           const arma::mat& subgradient,
                           const arma::umat& neighbour,
                           const std::size_t& disturb_index);

// Perform bellman recursion using nearest neighbours + row-rearrange
Rcpp::List BellmanAccelerated(Rcpp::NumericMatrix grid_,
                              Rcpp::NumericVector reward_,
                              Rcpp::NumericVector control_,
                              Rcpp::NumericVector disturb_,
                              Rcpp::NumericVector weight,
                              int n_neighbour,
                              Rcpp::Function Neighbour_);

// Perform bellman recursion using row rearrangement + nearest neighbours
arma::mat ExpectedAccelerated(Rcpp::NumericMatrix grid_,
                              Rcpp::NumericMatrix value_,
                              Rcpp::NumericVector disturb_,
                              Rcpp::NumericVector weight,
                              int n_neighbour,
                              Rcpp::Function Neighbour_);


#endif  // INST_INCLUDE_ACCELERATED_H_
