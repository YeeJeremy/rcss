// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for slow methods
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_SLOW_H_
#define INST_INCLUDE_SLOW_H_

#include <RcppArmadillo.h>

// Finds the maximising subgradient
arma::mat Optimal(const arma::mat& grid, const arma::mat& subgradient);

// Perform bellman recursion using row rearrangement
Rcpp::List Bellman(Rcpp::NumericMatrix grid_,
                   Rcpp::NumericVector reward_,
                   Rcpp::NumericVector control_,
                   Rcpp::NumericVector disturb_,
                   Rcpp::NumericVector weight);

// Expected value using row rearrangement
arma::mat Expected(Rcpp::NumericMatrix grid_,
                   Rcpp::NumericMatrix value_,
                   Rcpp::NumericVector disturb_,
                   Rcpp::NumericVector weight);

#endif  // INST_INCLUDE_SLOW_H_
