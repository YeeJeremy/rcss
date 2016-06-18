// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Determining the optimal action and corresponding subgradient
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif
#include "inst/include/bellman.h"

void BellmanOptimal(const arma::mat& grid,
                    const arma::imat& control,
                    arma::cube *value,
                    const arma::mat& reward,
                    arma::cube *cont,
                    arma::ucube *action,
                    const int dec) {
  const std::size_t n_pos = control.n_rows;
  const std::size_t n_action = control.n_cols;
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  arma::cube value_temp(n_grid, n_action, n_pos);
  arma::cube subgrad(n_grid, n_action * n_dim, n_pos);
  std::size_t p, a, i;
  arma::uword best;
#pragma omp parallel
  {
#pragma omp for private(p, a, i)
    for (p = 0; p < n_pos; p++) {
      for (a = 0; a < n_action; a++) {
        i = (n_action * dec + a) * (n_dim * n_pos);
        subgrad.slice(p).cols(n_dim * a, n_dim * (a + 1) - 1) =
            reward.cols(i + p * n_dim, i + (p + 1) * n_dim - 1)
            + cont->slice(dec).cols(n_dim * (control(p, a) - 1),
                                    n_dim * control(p, a) - 1);
        value_temp.slice(p).col(a)
            = arma::sum(subgrad.slice(p)
                        .cols(n_dim * a, n_dim * (a + 1) - 1) % grid, 1);
      }
    }
#pragma omp for private(p, a, best)
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


