// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for fast methods
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_FAST_H_
#define INST_INCLUDE_FAST_H_

#include <RcppArmadillo.h>

// Block diagonal matrix
arma::mat BlockDiag(const arma::mat& input, std::size_t n_repeat);

// Smoothing by using nearest neighbours
arma::mat Smooth(const arma::mat& grid,
                 const arma::mat& subgradient,
                 const arma::umat& smooth_neighbour);

// Bellman recursion using nearest neighbours
Rcpp::List FastBellman(Rcpp::NumericMatrix grid_,
                       Rcpp::NumericVector reward_,
                       Rcpp::NumericVector control_,
                       Rcpp::IntegerMatrix r_index_,
                       Rcpp::NumericVector disturb_,
                       Rcpp::NumericVector weight,
                       Rcpp::Function Neighbour_,
                       int n_smooth,
                       Rcpp::Function SmoothNeighbour_);

// Fast expected value function using nearest neighbours
arma::mat FastExpected(Rcpp::NumericMatrix grid_,
                       Rcpp::NumericMatrix value_,
                       Rcpp::IntegerMatrix r_index_,
                       Rcpp::NumericVector disturb_,
                       Rcpp::NumericVector weight,
                       Rcpp::Function Neighbour_,
                       int n_smooth,
                       Rcpp::Function SmoothNeighbour_);

#endif  // INST_INCLUDE_FAST_H_
