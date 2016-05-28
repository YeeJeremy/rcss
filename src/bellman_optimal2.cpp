// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Determining the optimal action and corresponding subgradient
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include "inst/include/bellman.h"

void BellmanOptimal2(const arma::mat& grid,
                     const arma::cube& control2,
                     arma::cube *value,
                     const arma::mat& reward,
                     arma::cube *cont,
                     arma::ucube *action,
                     const int dec) {
  const std::size_t n_pos = control2.n_rows;
  const std::size_t n_action = control2.n_cols;
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  arma::cube value_temp(n_grid, n_action, n_pos);
  arma::cube subgrad(n_grid, n_action * n_dim, n_pos);
  std::size_t p, a, i, j;
  arma::uword best;
#pragma omp parallel
  {
#pragma omp for private(p, a, j, i)
    for (p = 0; p < n_pos; p++) {
      for (a = 0; a < n_action; a++) {
        i = (n_action * dec + a) * (n_dim * n_pos);
        subgrad.slice(p).cols(n_dim * a, n_dim * (a + 1) - 1) =
            reward.cols(i + p * n_dim, i + (p + 1) * n_dim - 1);
        for (j = 0; j < n_pos; j++) {
          subgrad.slice(p).cols(n_dim * a, n_dim * (a + 1) - 1) +=
              control2(p, a, j) *
              cont->slice(dec).cols(n_dim * j, n_dim * (j + 1) - 1);
        }
        value_temp.slice(p).col(a)
            = arma::sum(subgrad.slice(p)
                        .cols(n_dim * a, n_dim * (a + 1) - 1) % grid, 1);
      }
    }
#pragma omp parallel for private(p, a, best)
    for (p = 0; p < n_pos; p++) {
      for (a = 0; a < n_grid; a++) {
        value_temp.slice(p).row(a).max(best);
        action->at(a, p, dec) = best;
        value->slice(dec).cols(n_dim * p, n_dim * (p + 1) - 1).row(a) =
            subgrad.slice(p).cols(n_dim * best, n_dim * (best + 1) - 1).row(a);
      }
    }
  }
}


