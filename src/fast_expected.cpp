// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Expected value using the consitional expectation matrices
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/fast.h"

// Bellman recursion using nearest neighbours
//[[Rcpp::export]]
arma::mat FastExpected(const arma::mat& grid,
                       const arma::mat& value,
                       const arma::umat& r_index,
                       const arma::cube& disturb,
                       const arma::vec& weight,
                       const std::size_t& n_smooth) {
  // Parameters
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const std::size_t n_disturb = disturb.n_slices;
  const std::size_t n_perm = r_index.n_rows + 1;
  // Construct the constant and permutation matrices
  arma::mat constant(n_dim, n_dim);
  arma::cube perm(n_grid, n_grid, n_perm, arma::fill::zeros);
  ExpectMat(constant, perm, grid, r_index, disturb, weight);
  // Finding the nearest neighbours for smoothing (if selected)
  arma::umat smooth_neighbour(n_grid, n_smooth);
  if (n_smooth > 1) {
    smooth_neighbour = arma::conv_to<arma::umat>::from(rflann::FastKDNeighbour(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)),
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)), n_smooth)) - 1;
  }
  // Computing the continuation value function
  arma::mat continuation(n_grid, n_dim);
  continuation = perm.slice(0) * value * constant;
  for (std::size_t ii = 0; ii < (n_perm - 1); ii++) {
    continuation.col(r_index(ii, 1) - 1) += perm.slice(ii + 1) *
        value.col(r_index(ii, 0) - 1);
  }
  // Smooth the continuation value functions (if selected)
  if (n_smooth > 1) {
    continuation = Smooth(grid, continuation, smooth_neighbour);
  }
  return continuation;
}
