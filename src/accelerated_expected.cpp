// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Expected value using row rearrange + k nearest neighbour
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include <rflann.h>
#include "inst/include/accelerated.h"

// Perform bellman recursion using row rearrangement + nearest neighbours
//[[Rcpp::export]]
arma::mat AcceleratedExpected(const arma::mat& grid,
                              const arma::mat& value,
                              const arma::cube& disturb,
                              const arma::vec& weight,
                              const std::size_t& n_neighbour) {
  // Passing R objects to C++
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const std::size_t n_disturb = disturb.n_slices;
  // Finding the relevant nearest neighbours for each point
  arma::umat neighbour(n_grid * n_disturb, n_neighbour);
  {
    // Disturbed grids
    arma::mat disturb_grid(n_grid * n_disturb, n_dim);
#pragma omp parallel for
    for (std::size_t dd = 0; dd < n_disturb; dd++) {
      disturb_grid.rows(n_grid * dd, n_grid * (dd + 1) - 1) =
          grid * disturb.slice(dd).t();
    }
    // K-nearest neighbours
    neighbour = arma::conv_to<arma::umat>::from(rflann::FastKDNeighbour(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(disturb_grid)),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)), n_neighbour)) - 1;
  }
  // Compute the expected value function
  arma::mat continuation(n_grid, n_dim, arma::fill::zeros);
  arma::mat d_value(n_grid, n_dim);
  for (std::size_t dd = 0; dd < n_disturb; dd++) {
      d_value = value * disturb.slice(dd);
      continuation += weight(dd) *
          OptimalNeighbour(grid, d_value, neighbour.rows(n_grid * dd, n_grid * (dd + 1) - 1), dd);
  }
  return continuation;
}
