// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Determining the optimal action and corresponding subgradient
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include "inst/include/bellman.h"

// Bellman optimal for full control case
void BellmanOptimal(const arma::mat& grid,
                    const arma::imat& control,
                    arma::cube& value,
                    const arma::cube& reward,
                    arma::cube& cont,
                    const int& dec) {
  const std::size_t n_pos = control.n_rows;
  const std::size_t n_action = control.n_cols;
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  arma::cube value_temp(n_grid, n_action, n_pos);
  arma::cube subgrad(n_grid, n_dim * n_action, n_pos);
  std::size_t ii;
  arma::uword best;
#pragma omp parallel
  {
#pragma omp for private(ii)
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t aa = 0; aa < n_action; aa++) {
        ii = n_dim * n_action * pp + n_dim * aa;
        subgrad.slice(pp).cols(n_dim * aa, n_dim * (aa + 1) - 1) =
            reward.slice(dec).cols(ii, ii + n_dim - 1)
            + cont.slice(dec).cols(n_dim * (control(pp, aa) - 1),
                                   n_dim * control(pp, aa) - 1);
        value_temp.slice(pp).col(aa)
            = arma::sum(subgrad.slice(pp)
                        .cols(n_dim * aa, n_dim * (aa + 1) - 1) % grid, 1);
      }
    }
#pragma omp for private(best)
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t gg = 0; gg < n_grid; gg++) {
        value_temp.slice(pp).row(gg).max(best);
        value.slice(dec).cols(n_dim * pp, n_dim * (pp + 1) - 1).row(gg) =
            subgrad.slice(pp).cols(n_dim * best, n_dim * (best + 1) - 1).row(gg);
      }
    }
  }
}

// Bellman optimal for partial control case
void BellmanOptimal2(const arma::mat& grid,
                     const arma::cube& control2,
                     arma::cube& value,
                     const arma::cube& reward,
                     arma::cube& cont,
                     const int& dec) {
  const std::size_t n_pos = control2.n_rows;
  const std::size_t n_action = control2.n_cols;
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  arma::cube value_temp(n_grid, n_action, n_pos);
  arma::cube subgrad(n_grid, n_action * n_dim, n_pos);
  std::size_t ii;
  arma::uword best;
#pragma omp parallel
  {
#pragma omp for private(ii)
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t aa = 0; aa < n_action; aa++) {
        ii = n_dim * n_action * pp + n_dim * aa;
        subgrad.slice(pp).cols(n_dim * aa, n_dim * (aa + 1) - 1) =
            reward.slice(dec).cols(ii, ii + n_dim - 1);
        for (std::size_t jj = 0; jj < n_pos; jj++) {
          subgrad.slice(pp).cols(n_dim * aa, n_dim * (aa + 1) - 1) +=
              control2(pp, aa, jj) *
              cont.slice(dec).cols(n_dim * jj, n_dim * (jj + 1) - 1);
        }
        value_temp.slice(pp).col(aa)
            = arma::sum(subgrad.slice(pp)
                        .cols(n_dim * aa, n_dim * (aa + 1) - 1) % grid, 1);
      }
    }
#pragma omp for private(best)
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t gg = 0; gg < n_grid; gg++) {
        value_temp.slice(pp).row(gg).max(best);
        value.slice(dec).cols(n_dim * pp, n_dim * (pp + 1) - 1).row(gg) =
            subgrad.slice(pp).cols(n_dim * best, n_dim * (best + 1) - 1).row(gg);
      }
    }
  }
}
