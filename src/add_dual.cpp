// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Additive duals using the row rearrangement operator
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

// Finds the maximum value
arma::vec OptimalValue(const arma::mat& state,
                       const arma::mat& subgradient) {
  const arma::mat t_state = state.t();
  const std::size_t n_state = state.n_rows;
  arma::vec optimal(n_state);
#pragma omp parallel for
  for (std::size_t ii = 0; ii < n_state; ii++) {
    optimal(ii) = (subgradient * t_state.col(ii)).max();
  }
  return optimal;
}

// Calculate the martingale increments using the row rearrangement
//[[Rcpp::export]]
arma::cube AddDual(const arma::cube& path,
                   Rcpp::NumericVector subsim_,
                   const arma::vec& weight,
                   Rcpp::NumericVector value_,
                   Rcpp::Function Scrap_) {
  // R objects to C++
  const std::size_t n_path = path.n_rows;
  const arma::ivec v_dims = value_.attr("dim");
  const std::size_t n_grid = v_dims(0);
  const std::size_t n_dim = v_dims(1);
  const std::size_t n_pos = v_dims(2);
  const std::size_t n_dec = v_dims(3);
  const arma::cube value(value_.begin(), n_grid, n_dim * n_pos, n_dec, false);
  const arma::ivec s_dims = subsim_.attr("dim");
  const std::size_t n_subsim = s_dims(2);
  const arma::cube subsim(subsim_.begin(), n_dim, n_dim * n_subsim * n_path, n_dec - 1, false);
  // Duals
  arma::cube mart(n_path, n_pos, n_dec - 1);
  arma::mat temp_state(n_subsim * n_path, n_dim);
  arma::mat fitted(n_grid, n_dim);
  std::size_t ll;
  Rcpp::Rcout << "Additive duals at dec: ";
  // Find averaged value
  for (std::size_t tt = 0; tt < (n_dec - 2); tt++) {
    Rcpp::Rcout << tt << "...";
    // 1 step subsimulation
#pragma omp parallel for private(ll)
    for (std::size_t ii = 0; ii < n_path; ii++) {
      for (std::size_t ss = 0; ss < n_subsim; ss++) {
        ll = n_subsim * ii + ss;
        temp_state.row(ll) = weight(ss) * path.slice(tt).row(ii) *
            arma::trans(subsim.slice(tt).cols(n_dim * ll, n_dim * (ll + 1) - 1));       
      }
    }
    // Averaging
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      fitted = value.slice(tt + 1).cols(n_dim * pp, n_dim * (pp + 1) - 1);
      mart.slice(tt).col(pp) = arma::conv_to<arma::vec>::from(arma::sum(arma::reshape(
          OptimalValue(temp_state, fitted), n_subsim, n_path)));
      // Subtract the path realisation
      mart.slice(tt).col(pp) -= OptimalValue(path.slice(tt + 1), fitted);
    }
  }
  // Scrap value
  Rcpp::Rcout << n_dec - 1 << "...";
  // 1 step subsimulation
#pragma omp parallel for private(ll)
  for (std::size_t ii = 0; ii < n_path; ii++) {
    for (std::size_t ss = 0; ss < n_subsim; ss++) {
      ll = n_subsim * ii + ss;
      temp_state.row(n_path * ss + ii) = path.slice(n_dec - 2).row(ii) *
          arma::trans(subsim.slice(n_dec - 2).cols(n_dim * ll, n_dim * (ll + 1) - 1));
    }
  }
  // Averaging
  arma::mat subsim_scrap(n_subsim * n_path, n_pos);
  subsim_scrap = Rcpp::as<arma::mat>(Scrap_(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(temp_state))));
  arma::mat scrap(n_path, n_pos);
  scrap = Rcpp::as<arma::mat>(Scrap_(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(n_dec - 1)))));
  for (std::size_t pp = 0; pp < n_pos; pp++) {
    mart.slice(n_dec - 2).col(pp) =
        arma::reshape(subsim_scrap.col(pp), n_path, n_subsim) * weight;
    // Subtract the path realisation
    mart.slice(n_dec - 2).col(pp) -= scrap.col(pp);
  }
  Rcpp::Rcout << "done\n";
  return mart;
}
