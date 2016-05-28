// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Fast methods for FastBellman and FastExpected
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include "inst/include/fast.h"

// Block diagonal matrix
arma::mat BlockDiag(const arma::mat& input, std::size_t n_repeat) {
  const std::size_t n_dim = input.n_rows;
  arma::mat block_diag(n_dim * n_repeat, n_dim * n_repeat);
  block_diag.fill(0);
  for (std::size_t i = 0; i < n_repeat; i++) {
    block_diag(arma::span(i * n_dim, (i + 1) * n_dim - 1),
               arma::span(i * n_dim, (i + 1) * n_dim - 1)) = input;
  }
  return block_diag;
}

// Smoothing by using nearest neighbours
arma::mat Smooth(const arma::mat& grid,
                 const arma::mat& subgradient,
                 const arma::umat& smooth_neighbour) {
  const arma::mat t_grid = grid.t();
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  arma::mat optimal(n_grid, n_dim);
  const std::size_t n_neighbour = smooth_neighbour.n_rows;
  arma::uword best;
  std::size_t i;
  arma::uvec near(n_neighbour);
#pragma omp parallel for private(i, best, near)
  for (i = 0; i < n_grid; i++) {
    near = smooth_neighbour.col(i);
    (subgradient.rows(near) * t_grid.col(i)).max(best);
    optimal.row(i) = subgradient.row(near(best));
  }
  return optimal;
}
