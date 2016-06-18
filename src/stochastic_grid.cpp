// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Generates a stochastic grid using k-means clustering
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif
#include <RcppArmadillo.h>
#include <Rcpp.h>

arma::mat StochasticGrid(Rcpp::NumericVector start_,
                         Rcpp::NumericVector disturb_,
                         int n_grid,
                         int max_iter,
                         bool warning) {
  // R objects to C++
  const arma::vec start(start_.begin(), start_.length(), false);
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_dim = d_dims(0);
  const std::size_t n_dec = d_dims(2) + 1;
  const std::size_t n_path = d_dims(3);
  const arma::mat disturb(disturb_.begin(), n_dim, n_dim * n_path * (n_dec-1), false);
  arma::mat grid(n_grid, n_dim);
  arma::mat temp_grid(n_grid - 1, n_dim);
  // Simulating the sample paths
  arma::mat path(n_dim, (n_dec - 1) * n_path);
  std::size_t i, j, k;
#pragma omp parallel for private(i, j, k)
  for (i = 0; i < n_path; i++) {
    k = i * (n_dec - 1);
    path.col(i * (n_dec - 1)) =
        disturb.cols(k * n_dim, (k + 1) * n_dim - 1) * start;
    for (j = 1; j < (n_dec - 1); j++) {
      k = i * (n_dec - 1) + j;
      path.col(i * (n_dec - 1) + j) =
          disturb.cols(k * n_dim, (k + 1) * n_dim - 1) * path.col(k - 1);
    }
  }
  // Perform the k-means algorithm
  arma::gmm_diag model;
  model.learn(path.rows(1, n_dim - 1), n_grid - 1, arma::eucl_dist,
              arma::random_subset, max_iter, 0, 1e-10, warning);
  temp_grid.cols(1, n_dim - 1) = arma::trans(model.means);
  temp_grid.col(0).fill(1.0);
  grid.row(0) = arma::conv_to<arma::rowvec>::from(start);
  grid.rows(1, n_grid - 1) = temp_grid;
  return grid;
}

// Export function to R
RcppExport SEXP rcss_StochasticGrid(SEXP startSEXP,
                                    SEXP disturbSEXP,
                                    SEXP n_gridSEXP,
                                    SEXP max_iterSEXP,
                                    SEXP warningSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        start(startSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        disturb(disturbSEXP);
    Rcpp::traits::input_parameter<int>::type
        n_grid(n_gridSEXP);
    Rcpp::traits::input_parameter<int>::type
        max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter<bool>::type
        warning(warningSEXP);
    __result = Rcpp::wrap(StochasticGrid(start, disturb, n_grid, max_iter,
                                         warning));
    return __result;
END_RCPP
}
