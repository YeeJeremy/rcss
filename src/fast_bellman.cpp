// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Performs the fast bellman recursion
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/bellman.h"
#include "inst/include/fast.h"
#include <Rcpp.h>

// Bellman recursion using nearest neighbours
Rcpp::List FastBellman(Rcpp::NumericMatrix grid_,
                       Rcpp::NumericVector reward_,
                       Rcpp::NumericVector control_,
                       Rcpp::IntegerMatrix r_index_,
                       Rcpp::NumericVector disturb_,
                       Rcpp::NumericVector weight,
                       Rcpp::Function Neighbour_,
                       int n_smooth,
                       Rcpp::Function SmoothNeighbour_) {
  // R objects to C++
  const std::size_t n_grid = grid_.nrow();
  const std::size_t n_dim = grid_.ncol();
  const arma::mat grid(grid_.begin(), n_grid, n_dim, false);
  const arma::ivec r_dims = reward_.attr("dim");
  const std::size_t n_pos = r_dims(2);
  const std::size_t n_action = r_dims(3);
  const std::size_t n_dec = r_dims(4);
  const arma::mat
      reward(reward_.begin(), n_grid, n_dim * n_pos * n_action * n_dec, false);
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
  const std::size_t n_perm = r_index_.nrow() + 1;
  const arma::imat r_index(r_index_.begin(), n_perm - 1, 2, false);
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_disturb = d_dims(2);
  const arma::mat disturb(disturb_.begin(), n_dim, n_dim * n_disturb, false);
  // Construct the constant and permutation matrices
  arma::mat constant(n_dim, n_dim);
  arma::cube perm(n_grid, n_grid, n_perm, arma::fill::zeros);
  {
  // Disturbed grids and nearest neighbours
    arma::uvec neighbour(n_grid * n_disturb);
    {
      arma::mat disturb_grid(n_grid * n_disturb, n_dim);
      // std::size_t d;
      // #pragma omp parallel for private(d) // doesnt really make it faster
      for (std::size_t d = 0; d < n_disturb; d++) {
        disturb_grid.rows(n_grid * d, n_grid * (d + 1) - 1) = grid *
            arma::trans(disturb.cols(d * n_dim, (d + 1) * n_dim - 1));
      }
      Rcpp::IntegerVector r_neighbour(n_grid * n_disturb);
      r_neighbour = Neighbour_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(disturb_grid)),
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)));
      neighbour = Rcpp::as<arma::uvec>(r_neighbour) - 1;  // R index to C++
    }
    // Constant matrix
    constant = disturb.cols(0, n_dim - 1);
    for (std::size_t i = 0; i < (n_perm - 1); i++) {
      constant(r_index(i, 0) - 1, r_index(i, 1) - 1) = 0;
    }
    // Conditional expectation operator
    for (std::size_t i = 0; i < n_disturb; i++) {
      for (std::size_t j = 0; j < n_grid; j++) {
        perm(j, neighbour(i * n_grid + j), 0) += weight[i];
        for (std::size_t k = 1; k < n_perm; k++) {
          perm(j, neighbour(i * n_grid + j), k) += weight[i] *
              disturb(r_index(k - 1, 0) - 1, r_index(k - 1, 1) - 1 + i * n_dim);
        }
      }
    }
  }
  // Finding the nearest neighbours for smoothing (if selected)
  arma::umat smooth_neighbour(n_smooth, n_grid);
  if (n_smooth > 1) {
    arma::umat temp_neighbour(n_grid, n_smooth);
    Rcpp::IntegerMatrix r_neighbour(n_grid, n_smooth);
    r_neighbour = SmoothNeighbour_(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)));
    temp_neighbour = Rcpp::as<arma::umat>(r_neighbour) - 1;  // R index to C++
    smooth_neighbour = temp_neighbour.t();
  }
  // Bellman recursion
  arma::cube value(n_grid, n_dim * n_pos, n_dec);
  arma::ucube action(n_grid, n_pos, n_dec);
  arma::cube cont(n_grid, n_dim * n_pos, n_dec, arma::fill::zeros);
  // Initialise with terminal time
  Rcpp::Rcout << "At dec: " << n_dec - 1;
  if (full_control) {
    BellmanOptimal(grid, control, &value, reward, &cont, &action, n_dec - 1);
  } else {
    BellmanOptimal2(grid, control2, &value, reward, &cont, &action, n_dec - 1);
  }
  Rcpp::Rcout << "...";
  const arma::uvec
      seq = arma::linspace<arma::uvec>(0, (n_pos - 1) * n_dim, n_pos);
  arma::mat block_diag(n_dim * n_pos, n_dim * n_pos);
  block_diag = BlockDiag(constant, n_pos);
  // Backward induction
  for (int i = (n_dec - 2); i >= 0; i--) {
    Rcpp::Rcout << i << ".";
    // Approximating the continuation value
    cont.slice(i) = perm.slice(0) * value.slice(i + 1) * block_diag;
    Rcpp::Rcout << ".";
    for (std::size_t j = 0; j < (n_perm - 1); j++) {
      cont.slice(i).cols(seq + r_index(j, 1) - 1) =
          cont.slice(i).cols(seq + r_index(j, 1) - 1) + perm.slice(j + 1) *
          value.slice(i + 1).cols(seq + r_index(j, 0) - 1);
    }
    Rcpp::Rcout << ".";
    // Smooth the continuation value functions (if selected)
    if (n_smooth > 1) {
      for (std::size_t k = 0; k < n_pos; k++) {
        cont.slice(i).cols(k * n_dim, (k + 1) * n_dim - 1) =
            Smooth(grid, cont.slice(i).cols(k * n_dim, (k + 1) * n_dim - 1),
                   smooth_neighbour);
      }
    }
    // Find optimal action and the corresponding value function
    if (full_control) {
      BellmanOptimal(grid, control, &value, reward, &cont, &action, i);
    } else {
      BellmanOptimal2(grid, control2, &value, reward, &cont, &action, i);
    }
  }
  action += 1;  // C++ indexing to R indexing
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("expected") = cont,
                            Rcpp::Named("action") = action);
}

// Export function to R
RcppExport SEXP rcss_FastBellman(SEXP gridSEXP,
                                 SEXP rewardSEXP,
                                 SEXP controlSEXP,
                                 SEXP r_indexSEXP,
                                 SEXP disturbSEXP,
                                 SEXP weightSEXP,
                                 SEXP NeighbourSEXP,
                                 SEXP n_smoothSEXP,
                                 SEXP SmoothNeighbourSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericMatrix>::type
        grid(gridSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        reward(rewardSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        control(controlSEXP);
    Rcpp::traits::input_parameter<Rcpp::IntegerMatrix>::type
        r_index(r_indexSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        disturb(disturbSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        weight(weightSEXP);
    Rcpp::traits::input_parameter<Rcpp::Function>::type
        Neighbour(NeighbourSEXP);
    Rcpp::traits::input_parameter<int>::type
        n_smooth(n_smoothSEXP);
    Rcpp::traits::input_parameter<Rcpp::Function>::type
        SmoothNeighbour(SmoothNeighbourSEXP);
    __result = Rcpp::wrap(FastBellman(grid, reward, control, r_index, disturb,
                                      weight, Neighbour, n_smooth,
                                      SmoothNeighbour));
    return __result;
END_RCPP
}
