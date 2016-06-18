// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Simulating paths using user supplied disturbances
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif
#include <RcppArmadillo.h>
#include <Rcpp.h>

arma::cube Path(Rcpp::NumericVector start_, Rcpp::NumericVector disturb_) {
  // R objects to C++
  const arma::vec start(start_.begin(), start_.length(), false);
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_dim = d_dims(0);
  const std::size_t n_dec = d_dims(2) + 1;
  const std::size_t n_path = d_dims(3);
  const arma::mat disturb(disturb_.begin(), n_dim, n_dim * (n_dec - 1) * n_path,
			  false);
  // Simulating the sample paths
  arma::cube path(n_dec, n_path, n_dim);
  arma::vec state(n_dim);
  std::size_t i, j, k;
#pragma omp parallel for private(i, j, k, state)
  for (i = 0; i < n_path; i++) {
    path.tube(0, i) = start;
    for (j = 1; j < n_dec; j++) {
      k = i * (n_dec - 1) + j - 1;
      state = path.tube(j - 1, i);
      path.tube(j, i) = disturb.cols(k * n_dim, (k + 1) * n_dim - 1) * state;
    }
  }
  return path;
}

// Export to R
RcppExport SEXP rcss_Path(SEXP startSEXP,
                          SEXP disturbSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        start(startSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        disturb(disturbSEXP);
    __result = Rcpp::wrap(Path(start, disturb));
    return __result;
END_RCPP
}
