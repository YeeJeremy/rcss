// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Fast methods for FastBellman and FastExpected
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include "inst/include/fast.h"

// Block diagonal matrix
arma::mat BlockDiag(const arma::mat& input,
                    std::size_t n_repeat) {
  const std::size_t n_dim = input.n_rows;
  arma::mat block_diag(n_dim * n_repeat, n_dim * n_repeat);
  block_diag.fill(0);
  for (std::size_t ii = 0; ii < n_repeat; ii++) {
    block_diag(arma::span(ii * n_dim, (ii + 1) * n_dim - 1),
               arma::span(ii * n_dim, (ii + 1) * n_dim - 1)) = input;
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
  const std::size_t n_neighbour = smooth_neighbour.n_cols;
  arma::uword best;
  arma::umat neighbour = arma::conv_to<arma::umat>::from(smooth_neighbour);
  arma::urowvec near(n_neighbour);
#pragma omp parallel for private(best, near)
  for (std::size_t gg = 0; gg < n_grid; gg++) {
    near = neighbour.row(gg);
    (subgradient.rows(near) * t_grid.col(gg)).max(best);
    optimal.row(gg) = subgradient.row(near(best));
  }
  return optimal;
}

// Generate the conditional expectation matrices
void ExpectMat(arma::mat& constant,
               arma::cube& perm,
               const arma::mat& grid,
               const arma::umat& r_index,
               const arma::cube& disturb,
               const arma::vec& weight) {
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const std::size_t n_perm = r_index.n_rows + 1;
  const std::size_t n_disturb = disturb.n_slices;
  // Disturbed grids and nearest neighbours
  arma::umat neighbour(n_grid * n_disturb, 1);
  {
    arma::mat disturb_grid(n_grid * n_disturb, n_dim);
    for (std::size_t dd = 0; dd < n_disturb; dd++) {
      disturb_grid.rows(n_grid * dd, n_grid * (dd + 1) - 1) =
          grid * arma::trans(disturb.slice(dd));
    }
    neighbour = arma::conv_to<arma::umat>::from(rflann::FastKDNeighbour(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(disturb_grid.cols(1, n_dim - 1))),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid.cols(1, n_dim - 1))),
        1)) - 1;
  }
  // Constant matrix
  constant = disturb.slice(0);
  for (std::size_t pp = 0; pp < (n_perm - 1); pp++) {
    constant(r_index(pp, 0) - 1, r_index(pp, 1) - 1) = 0;
  }
  // Conditional expectation operator
  std::size_t dd, gg, pp;
  for (dd = 0; dd < n_disturb; dd++) {
    for (gg = 0; gg < n_grid; gg++) {
      perm(gg, neighbour(n_grid * dd + gg), 0) += weight(dd);
      for (pp = 1; pp < n_perm; pp++) {
        perm(gg, neighbour(n_grid * dd + gg), pp) += weight(dd) *
            disturb(r_index(pp - 1, 0) - 1, r_index(pp - 1, 1) - 1, dd);
      }
    }
  }
}

