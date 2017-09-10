// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Bounds estimates using additive duals
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <RcppArmadillo.h>

//[[Rcpp::export]]
Rcpp::List AddDualBounds(const arma::cube& path,
                         Rcpp::NumericVector control_,
                         Rcpp::Function Reward_,
                         Rcpp::Function Scrap_,
                         const arma::cube& mart,
                         const arma::ucube& path_action) {
  // Extract parameters
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  const arma::ivec c_dims = control_.attr("dim");
  const std::size_t n_pos = c_dims(0);
  const std::size_t n_action = c_dims(1);
  arma::imat control;  // full control
  arma::cube control2;  // partial control
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
  // Initialise with scrap value
  arma::cube primals(n_path, n_pos, n_dec);
  primals.slice(n_dec - 1) = Rcpp::as<arma::mat>(
      Scrap_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(n_dec - 1)))));
  arma::cube duals = primals;
  // Perform the backward induction
  arma::uword policy;
  arma::cube reward(n_path, n_action, n_pos);
  if (full_control) {  // For the full control case
    arma::uword next;
    for (int tt = (n_dec - 2); tt >= 0; tt--) {
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
#pragma omp parallel for private(policy, next)
      for (std::size_t ii = 0; ii < n_path; ii++) {
        for (std::size_t pp = 0; pp < n_pos; pp++) {
          // Primal values
          policy = path_action(ii, pp, tt) - 1;  // R to C indexing
          next = control(pp, policy) - 1;
          primals(ii, pp, tt) = reward(ii, policy, pp) + mart(ii, next, tt)
              + primals(ii, next, tt + 1);
          // Dual values
          next = control(pp, 0) - 1;
          duals(ii, pp, tt) = reward(ii, 0, pp) + mart(ii, next, tt)
              + duals(ii, next, tt + 1);
          for (std::size_t aa = 1; aa < n_action; aa++) {
            next = control(pp, aa) - 1;
            duals(ii, pp, tt) = std::max(reward(ii, aa, pp) + mart(ii, next, tt)
                         + duals(ii, next, tt + 1), duals(ii, pp, tt));
          }
        }
      }
    }
  } else {  // Positions evolve randomly
    arma::rowvec mod(n_pos);
    arma::rowvec prob_weight(n_pos);
    for (int tt = (n_dec - 2); tt >= 0; tt--) {
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
#pragma omp parallel for private(policy, prob_weight, mod)
      for (std::size_t ii = 0; ii < n_path; ii++) {
        for (std::size_t pp = 0; pp < n_pos; pp++) {
          //  Primal values
          mod = primals.slice(tt + 1).row(ii) + mart.slice(tt).row(ii);
          policy = path_action(ii, pp, tt) - 1;
          prob_weight = control2.tube(pp, policy);
          primals(ii, pp, tt) =
              reward(ii, policy, pp) + arma::sum(prob_weight % mod);
          // Dual values
          mod = duals.slice(tt + 1).row(ii) + mart.slice(tt).row(ii);
          prob_weight = control2.tube(pp, 0);
          duals(ii, pp, tt) = reward(ii, 0, pp) + arma::sum(prob_weight % mod);
          for (std::size_t aa = 1; aa < n_action; aa++) {
            prob_weight = control2.tube(pp, aa);
            duals(ii, pp, tt) =
                std::max(reward(ii, aa, pp) + arma::sum(prob_weight % mod),
                         duals(ii, pp, tt));
          }
        }
      }
    }
  }
  return Rcpp::List::create(Rcpp::Named("primal") = primals,
                            Rcpp::Named("dual") = duals);
}
