// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for bellman optimal function
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_BELLMAN_H_
#define INST_INCLUDE_BELLMAN_H_

#include <RcppArmadillo.h>

// Determining the optimal action and corresponding subgradient
void BellmanOptimal(const arma::mat& grid,
                    const arma::imat& control,
                    arma::cube *value,
                    const arma::mat& reward,
                    arma::cube *cont,
                    arma::ucube *action,
                    const int dec);

// Determine the optimal action when position control non-deterministic
void BellmanOptimal2(const arma::mat& grid,
                     const arma::cube& control2,
                     arma::cube *value,
                     const arma::mat& reward,
                     arma::cube *cont,
                     arma::ucube *action,
                     const int dec);

// Perform bellman recursion using row rearrangement
Rcpp::List Bellman(Rcpp::NumericMatrix grid_,
                   Rcpp::NumericVector reward_,
                   Rcpp::NumericVector control_,
                   Rcpp::NumericVector disturb_,
                   Rcpp::NumericVector weight);

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

// Perform bellman recursion using nearest neighbours
Rcpp::List BellmanAccelerated(Rcpp::NumericMatrix grid_,
                              Rcpp::NumericVector reward_,
                              Rcpp::NumericVector control_,
                              Rcpp::NumericVector disturb_,
                              Rcpp::NumericVector weight,
                              int n_neighbour,
                              Rcpp::Function Neighbour_);

#endif  // INST_INCLUDE_BELLMAN_H_
