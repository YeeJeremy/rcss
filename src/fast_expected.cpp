// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Performs the fast bellman recursion
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/bellman.h"
#include "inst/include/fast.h"
#include <Rcpp.h>

// Bellman recursion using nearest neighbours
arma::mat FastExpected(Rcpp::NumericMatrix grid_,
                       Rcpp::NumericMatrix value_,
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
  const arma::mat value(value_.begin(), n_grid, n_dim, false);
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
      std::size_t d;
#pragma omp parallel for private(d)
      for (d = 0; d < n_disturb; d++) {
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
  arma::umat smooth_neighbour(n_smooth, n_grid * n_disturb);
  if (n_smooth > 1) {
    arma::umat temp_neighbour(n_grid, n_smooth);
    Rcpp::IntegerMatrix r_neighbour(n_grid, n_smooth);
    r_neighbour = SmoothNeighbour_(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)));
    temp_neighbour = Rcpp::as<arma::umat>(r_neighbour) - 1;  // R index to C++
    smooth_neighbour = temp_neighbour.t();
  }
  // Computing the continuation value function
  arma::mat continuation(n_grid, n_dim);
  continuation = perm.slice(0) * value * constant;
  for (std::size_t i = 0; i < (n_perm - 1); i++) {
    continuation.col(r_index(i, 1) - 1) += perm.slice(i + 1) *
        value.col(r_index(i, 0) - 1);
  }
  // Smooth the continuation value functions (if selected)
  if (n_smooth > 1) {
    continuation = Smooth(grid, continuation, smooth_neighbour);
  }
  return continuation;
}

// Export function to R
RcppExport SEXP rcss_FastExpected(SEXP gridSEXP,
                                  SEXP valueSEXP,
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
    Rcpp::traits::input_parameter<Rcpp::NumericMatrix>::type
        value(valueSEXP);
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
    __result = Rcpp::wrap(FastExpected(grid, value, r_index, disturb,
                                       weight, Neighbour, n_smooth,
                                       SmoothNeighbour));
    return __result;
END_RCPP
}
