// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Simulating paths using supplied disturbances
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

//[[Rcpp::export]]
arma::cube PathDisturb(const arma::vec& start,
                       Rcpp::NumericVector disturb_) {
  // R objects to C++
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_dim = d_dims(0);
  const std::size_t n_dec = d_dims(3) + 1;
  const std::size_t n_path = d_dims(2);
  const arma::cube disturb(disturb_.begin(), n_dim, n_dim * n_path, n_dec - 1,
			  false);
  // Simulating the sample paths
  arma::cube path(n_path, n_dim, n_dec);
  // Assign starting values
  for (std::size_t ii = 0; ii < n_dim; ii++) {
    path.slice(0).col(ii).fill(start(ii));
  }
  // Disturb the paths
#pragma omp parallel for
  for (std::size_t pp = 0; pp < n_path; pp++) {
    for (std::size_t tt = 1; tt < n_dec; tt++) {
      path.slice(tt).row(pp) = path.slice(tt - 1).row(pp) *
          arma::trans(disturb.slice(tt - 1).cols(n_dim * pp, n_dim * (pp + 1) - 1));
    }
  }
  return path;
}
