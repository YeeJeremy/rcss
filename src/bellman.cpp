// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Bellman recursion using row rearrangement
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include "inst/include/bellman.h"
#include "inst/include/slow.h"

#pragma omp declare reduction( + : arma::mat : omp_out += omp_in ) \
  initializer( omp_priv = omp_orig )

// Expected value using row rearrangement
//[[Rcpp::export]]
arma::mat Expected(const arma::mat& grid,
                   const arma::mat& value,
                   const arma::cube& disturb,
                   const arma::vec& weight) {
  // Passing R objects to C++
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const std::size_t n_disturb = disturb.n_slices;
  // Computing the continuation value function
  arma::mat continuation(n_grid, n_dim, arma::fill::zeros);
  arma::mat d_value(n_grid, n_dim);
#pragma omp parallel for private(d_value) reduction(+:continuation)
  for (std::size_t dd = 0; dd < n_disturb; dd++) {
    d_value = value * disturb.slice(dd);
    continuation += weight(dd) * Optimal(grid, d_value);
  }
  return continuation;
}

// Bellman recursion using row rearrangement
//[[Rcpp::export]]
Rcpp::List Bellman(const arma::mat& grid,
                   Rcpp::NumericVector reward_,
                   const arma::cube& scrap,
                   Rcpp::NumericVector control_,
                   const arma::cube& disturb,
                   const arma::vec& weight) {
  // Passing R objects to C++
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const arma::ivec r_dims = reward_.attr("dim");
  const std::size_t n_pos = r_dims(3);
  const std::size_t n_action = r_dims(2);
  const std::size_t n_dec = r_dims(4) + 1;
  const arma::cube
      reward(reward_.begin(), n_grid, n_dim * n_action * n_pos, n_dec - 1, false);
  const arma::ivec c_dims = control_.attr("dim");
  arma::cube control2;
  arma::imat control;
  bool full_control;
  if (c_dims.n_elem == 3) {
    full_control = false;
    arma::cube temp_control2(control_.begin(), n_pos, n_action, n_pos, false);
    control2 = temp_control2;
  } else {
    full_control = true;
    arma::mat temp_control(control_.begin(), n_pos, n_action, false);
    control = arma::conv_to<arma::imat>::from(temp_control);
  }
  const std::size_t n_disturb = disturb.n_slices;
  // Bellman recursion
  arma::cube value(n_grid, n_dim * n_pos, n_dec);
  arma::cube cont(n_grid, n_dim * n_pos, n_dec - 1, arma::fill::zeros);
  arma::mat d_value(n_grid, n_dim);
  Rcpp::Rcout << "At dec: " << n_dec - 1 << "...";
  for (std::size_t pp = 0; pp < n_pos; pp++) {
    value.slice(n_dec - 1).cols(n_dim * pp, n_dim * (pp + 1) - 1) =
        scrap.slice(pp);
  }
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << tt;
    // Approximating the continuation value
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      cont.slice(tt).cols(n_dim * pp, n_dim * (pp + 1) - 1) =
          Expected(grid,
                   value.slice(tt + 1).cols(pp * n_dim, n_dim * (pp + 1) - 1),
                   disturb, weight);
    }
    Rcpp::Rcout << "..";
    // Optimise value function
    if (full_control) {
      BellmanOptimal(grid, control, value, reward, cont, tt);
    } else {
      BellmanOptimal2(grid, control2, value, reward, cont, tt);
    }
    Rcpp::Rcout << ".";
  }
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("expected") = cont);
}
