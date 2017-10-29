// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Bellman recursion using row rearrange + k nearest neighbour
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include <rflann.h>
#include "inst/include/bellman.h"
#include "inst/include/accelerated.h"

#pragma omp declare reduction( + : arma::mat : omp_out += omp_in )    \
  initializer( omp_priv = omp_orig )

// Perform bellman recursion using nearest neighbours
//[[Rcpp::export]]
Rcpp::List AcceleratedBellman(const arma::mat& grid,
                              Rcpp::NumericVector reward_,
                              const arma::cube& scrap,
                              Rcpp::NumericVector control_,
                              const arma::cube& disturb,
                              const arma::vec& weight,
                              const std::size_t& n_neighbour) {
  // Passing R objects to C++
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const arma::ivec r_dims = reward_.attr("dim");
  const std::size_t n_pos = r_dims(3);
  const std::size_t n_action = r_dims(2);
  const std::size_t n_dec = r_dims(4) + 1;
  const arma::cube
      reward(reward_.begin(), n_grid, n_dim * n_pos * n_action, n_dec - 1, false);
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
  // Finding the relevant nearest neighbours for each point
  arma::umat neighbour(n_grid * n_disturb, n_neighbour);
  {
    // Disturbed grids
    arma::mat disturb_grid(n_grid * n_disturb, n_dim);
#pragma omp parallel for
    for (std::size_t dd = 0; dd < n_disturb; dd++) {
      disturb_grid.rows(n_grid * dd, n_grid * (dd + 1) - 1) =
          grid * disturb.slice(dd).t();
    }
    // K-nearest neighbours
    neighbour = arma::conv_to<arma::umat>::from(rflann::FastKDNeighbour(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(disturb_grid)),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)), n_neighbour)) - 1;
  }
  // Bellman recursion
  arma::cube value(n_grid, n_dim * n_pos, n_dec);
  arma::cube cont(n_grid, n_dim * n_pos, n_dec - 1, arma::fill::zeros);
  // Initialise with terminal time
  Rcpp::Rcout << "At dec: " << n_dec - 1 << "...";
  for (std::size_t pp = 0; pp < n_pos; pp++) {
    value.slice(n_dec - 1).cols(n_dim * pp, n_dim * (pp + 1) - 1) =
        scrap.slice(pp);
  }
  // Backwards induction
  arma::mat d_value(n_grid, n_dim);
  arma::mat temp_cont(n_grid, n_dim * n_pos);
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << tt << ".";
    // Approximating the continuation value
    temp_cont.fill(0.);
#pragma omp parallel for private(d_value) reduction(+:temp_cont)
    for (std::size_t dd = 0; dd < n_disturb; dd++) {
      for (std::size_t pp = 0; pp < n_pos; pp++) {
        d_value = value.slice(tt + 1).cols(n_dim * pp, n_dim * (pp + 1) - 1)
            * disturb.slice(dd);
        temp_cont.cols(n_dim * pp, n_dim * (pp + 1) - 1) += weight(dd) *
            OptimalNeighbour(grid, d_value, neighbour.rows(n_grid * dd, n_grid * (dd + 1) - 1));
        //cont.slice(tt).cols(n_dim * pp, n_dim * (pp + 1) - 1) += weight(dd) *
        //    OptimalNeighbour(grid, d_value, neighbour.rows(n_grid * dd, n_grid * (dd + 1) - 1));
      }
    }
    cont.slice(tt) = temp_cont;
    Rcpp::Rcout << "..";
    // Optimise to find value function
    if (full_control) {
      BellmanOptimal(grid, control, value, reward, cont, tt);
    } else {
      BellmanOptimal2(grid, control2, value, reward, cont, tt);
    }
  }
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("expected") = cont);
}
