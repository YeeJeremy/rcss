// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Fast methods for FastBellman and FastExpected
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include "inst/include/accelerated.h"

// Finds the maximising subgradient using nearest neighbours + row-rearrange
arma::mat OptimalNeighbour(const arma::mat& grid,
                           const arma::mat& subgradient,
                           const arma::umat& neighbour,
                           const std::size_t& disturb_index) {
  const arma::mat t_grid = grid.t();
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  arma::mat optimal(n_grid, n_dim);
  const std::size_t n_neighbour = neighbour.n_rows;
  arma::uword best;
  std::size_t i;
  arma::uvec near(n_neighbour);
#pragma omp parallel for private(i, best, near)
  for (i = 0; i < n_grid; i++) {
    near = neighbour.col(disturb_index * n_grid + i);
    (subgradient.rows(near) * t_grid.col(i)).max(best);
    optimal.row(i) = subgradient.row(near(best));
  }
  return optimal;
}
