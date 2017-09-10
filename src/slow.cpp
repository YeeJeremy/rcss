// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Row rearrangement operator
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/slow.h"

// Finds the maximising subgradient
//[[Rcpp::export]]
arma::mat Optimal(const arma::mat& grid,
                  const arma::mat& subgradient) {
  const arma::mat t_grid = grid.t();
  const std::size_t n_grid = grid.n_rows;
  arma::mat compare(n_grid, n_grid);
  compare = subgradient * t_grid;
  // Maximising index in each col = max for each grid point
  arma::urowvec max_index(n_grid);
  max_index = arma::index_max(compare);
  arma::mat optimal(n_grid, grid.n_cols);
  for (std::size_t gg = 0; gg < n_grid; gg++) {
    optimal.row(gg) = subgradient.row(max_index(gg));
  }
  return optimal;
}
