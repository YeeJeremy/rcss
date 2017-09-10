// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Fast Bellman recursion
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/bellman.h"
#include "inst/include/fast.h"

// Bellman recursion using the conditional expectation matrices
//[[Rcpp::export]]
Rcpp::List FastBellman(const arma::mat& grid,
                       Rcpp::NumericVector reward_,
                       const arma::cube& scrap,
                       Rcpp::NumericVector control_,
                       const arma::umat& r_index,
                       const arma::cube& disturb,
                       const arma::vec& weight,
                       const std::size_t& n_smooth) {
  // Parameters
  const std::size_t n_grid = grid.n_rows;
  const std::size_t n_dim = grid.n_cols;
  const arma::ivec r_dims = reward_.attr("dim");
  const std::size_t n_pos = r_dims(2);
  const std::size_t n_action = r_dims(3);
  const std::size_t n_dec = r_dims(4) + 1;
  const arma::cube
      reward(reward_.begin(), n_grid, n_dim * n_action * n_pos, n_dec - 1, false);
  const arma::ivec c_dims = control_.attr("dim");
  arma::imat control;
  arma::cube control2;
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
  const std::size_t n_perm = r_index.n_rows + 1;
  const std::size_t n_disturb = disturb.n_slices;
  // Construct the conditional expectation matrices
  arma::mat constant(n_dim, n_dim);
  arma::cube perm(n_grid, n_grid, n_perm, arma::fill::zeros);
  ExpectMat(constant, perm, grid, r_index, disturb, weight);
  // Finding the nearest neighbours for smoothing (if selected)
  arma::umat smooth_neighbour(n_grid, n_smooth);
  if (n_smooth > 1) {
    smooth_neighbour = arma::conv_to<arma::umat>::from(rflann::FastKDNeighbour(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)),
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)), n_smooth)) - 1;
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
  // The following is for faster calculations
  const arma::uvec
      seq = arma::linspace<arma::uvec>(0, (n_pos - 1) * n_dim, n_pos);
  arma::mat block_diag(n_dim * n_pos, n_dim * n_pos);
  block_diag = BlockDiag(constant, n_pos);
  // Backward induction
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << tt << ".";
    // Approximating the continuation value
    cont.slice(tt) = perm.slice(0) * value.slice(tt + 1) * block_diag;
    Rcpp::Rcout << ".";
    for (std::size_t pp = 0; pp < (n_perm - 1); pp++) {
      cont.slice(tt).cols(seq + r_index(pp, 1) - 1) =
          cont.slice(tt).cols(seq + r_index(pp, 1) - 1) + perm.slice(pp + 1) *
          value.slice(tt + 1).cols(seq + r_index(pp, 0) - 1);
    }
    Rcpp::Rcout << ".";
    // Smooth the continuation value functions (if selected)
    if (n_smooth > 1) {
      for (std::size_t pp = 0; pp < n_pos; pp++) {
        cont.slice(tt).cols(pp * n_dim, (pp + 1) * n_dim - 1) =
            Smooth(grid, cont.slice(tt).cols(pp * n_dim, (pp + 1) * n_dim - 1),
                   smooth_neighbour);
      }
    }
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
