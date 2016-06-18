// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Slow methods for Bellman and Expected
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif
#include "inst/include/slow.h"

// Finds the maximising subgradient
arma::mat Optimal(const arma::mat& grid, const arma::mat& subgradient) {
  const arma::mat t_grid = grid.t();
  const std::size_t n_grid = grid.n_rows;
  arma::mat optimal(n_grid, grid.n_cols);
  arma::uword best;
  std::size_t i;
#pragma omp parallel for private(i, best)
  for (i = 0; i < n_grid; i++) {
    (subgradient * t_grid.col(i)).max(best);
    optimal.row(i) = subgradient.row(best);
  }
  return optimal;
}
