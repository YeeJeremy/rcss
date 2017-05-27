// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Performs the bellman recursion using row rearrangement
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/bellman.h"
#include "inst/include/slow.h"
#include <Rcpp.h>

// Perform bellman recursion using row rearrangement
//[[Rcpp::export]]
arma::mat Expected(Rcpp::NumericMatrix grid_,
                   Rcpp::NumericMatrix value_,
                   Rcpp::NumericVector disturb_,
                   Rcpp::NumericVector weight) {
  // Passing R objects to C++
  const std::size_t n_grid = grid_.nrow();
  const std::size_t n_dim = grid_.ncol();
  const arma::mat grid(grid_.begin(), n_grid, n_dim, false);
  const arma::mat value(value_.begin(), n_grid, n_dim, false);
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_disturb = d_dims(2);
  const arma::mat disturb(disturb_.begin(), n_dim, n_dim * n_disturb, false);
  // Computing the continuation value function
  arma::mat continuation(n_grid, n_dim, arma::fill::zeros);
  arma::mat d_value(n_grid, n_dim);
  for (std::size_t j = 0; j < n_disturb; j++) {
    d_value = value * disturb.cols(j * n_dim, (j + 1) * n_dim - 1);
    continuation += weight[j] * Optimal(grid, d_value);
  }
  return continuation;
}
