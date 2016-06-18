// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Performs the bellman recursion using row rearrange + k nearest neighbour
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/bellman.h"
#include "inst/include/accelerated.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <Rcpp.h>

// Perform bellman recursion using row rearrangement + nearest neighbours
arma::mat ExpectedAccelerated(Rcpp::NumericMatrix grid_,
                              Rcpp::NumericMatrix value_,
                              Rcpp::NumericVector disturb_,
                              Rcpp::NumericVector weight,
                              int n_neighbour,
                              Rcpp::Function Neighbour_) {
  // Passing R objects to C++
  const std::size_t n_grid = grid_.nrow();
  const std::size_t n_dim = grid_.ncol();
  const arma::mat grid(grid_.begin(), n_grid, n_dim, false);
  const arma::mat value(value_.begin(), n_grid, n_dim, false);
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_disturb = d_dims(2);
  const arma::mat disturb(disturb_.begin(), n_dim, n_dim * n_disturb, false);
  // Finding the relevant nearest neighbours for each point
  arma::umat neighbour(n_neighbour, n_grid * n_disturb);
  {
    // Disturbed grids
    arma::mat disturb_grid(n_grid * n_disturb, n_dim);
    std::size_t d;
#pragma omp parallel for private(d)
    for (d = 0; d < n_disturb; d++) {
      disturb_grid.rows(n_grid * d, n_grid * (d + 1) - 1) = grid *
          arma::trans(disturb.cols(d * n_dim, (d + 1) * n_dim - 1));
    }
    // Call nearest neighbour function
    Rcpp::IntegerMatrix r_neighbour(n_grid * n_disturb, n_neighbour);
    arma::umat temp_neighbour(n_grid * n_disturb, n_neighbour);
    r_neighbour = Neighbour_(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(disturb_grid)),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)));
    temp_neighbour = Rcpp::as<arma::umat>(r_neighbour) - 1;  // R index to C++
    neighbour = temp_neighbour.t();
  }
  // Compute the expected value function
  arma::mat continuation(n_grid, n_dim, arma::fill::zeros);
  arma::mat d_value(n_grid, n_dim);
  for (std::size_t j = 0; j < n_disturb; j++) {
      d_value = value * disturb.cols(j * n_dim, (j + 1) * n_dim - 1);
      continuation += weight[j] * OptimalNeighbour(grid, d_value, neighbour, j);
  }
  return continuation;
}

// Export to R
RcppExport SEXP rcss_ExpectedAccelerated(SEXP gridSEXP,
                                         SEXP valueSEXP,
                                         SEXP disturbSEXP,
                                         SEXP weightSEXP,
                                         SEXP n_neighbourSEXP,
                                         SEXP NeighbourSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericMatrix>::type
        grid(gridSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericMatrix>::type
        value(valueSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        disturb(disturbSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        weight(weightSEXP);
    Rcpp::traits::input_parameter<int>::type
        n_neighbour(n_neighbourSEXP);
    Rcpp::traits::input_parameter<Rcpp::Function>::type
        Neighbour(NeighbourSEXP);
    __result = Rcpp::wrap(ExpectedAccelerated(grid, value, disturb, weight,
                                              n_neighbour, Neighbour));
    return __result;
END_RCPP
}
