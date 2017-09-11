// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Row rearragement amongst the k nearest neighbours
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/accelerated.h"

// Maximising subgradient using nearest neighbours + row-rearrange
arma::mat OptimalNeighbour(const arma::mat& grid,
                           const arma::mat& subgradient,
                           const arma::umat& neighbour) {
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const std::size_t n_neighbour = neighbour.n_cols;
  arma::mat compare(n_grid, n_neighbour);
  arma::uvec cols_ind = arma::linspace<arma::uvec>(0, n_dim - 1, n_dim);
  for (std::size_t nn = 0; nn < n_neighbour; nn++) {
    compare.col(nn) = arma::sum(subgradient.submat(neighbour.col(nn), cols_ind)
                                % grid, 1);
  }
  arma::uvec max_index(n_grid);
  max_index = arma::index_max(compare, 1);
  arma::mat optimal(n_grid, n_dim);
  for (std::size_t gg = 0; gg < n_grid; gg++) {
    optimal.row(gg) = subgradient.row(neighbour(gg, max_index(gg)));
  }
  return optimal;
}
