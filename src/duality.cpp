// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Solution diagnostics for the CSS estimates using nearest neighbours
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <RcppArmadillo.h>
#include <Rcpp.h>

Rcpp::List Duality(Rcpp::NumericVector path_,
                   Rcpp::NumericVector control_,
                   Rcpp::Function Reward_,
                   Rcpp::NumericVector mart_,
                   Rcpp::IntegerVector path_action_) {
  // R objects into C++
  const arma::ivec p_dims = path_.attr("dim");
  const std::size_t n_dec = p_dims(0);
  const std::size_t n_path = p_dims(1);
  const std::size_t n_dim = p_dims(2);
  const arma::cube path(path_.begin(), n_dec, n_path, n_dim, false);
  const arma::ivec c_dims = control_.attr("dim");
  const std::size_t n_pos = c_dims(0);
  const std::size_t n_action = c_dims(1);
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
  const arma::ivec m_dims = mart_.attr("dim");
  std::size_t n_repeat = n_pos;
  std::size_t n_repeat2 = 1;
  if (!full_control) {
    n_repeat = n_action;
    n_repeat2 = n_pos;
  }
  const arma::cube mart(mart_.begin(), n_dec - 1, n_repeat, n_repeat2 * n_path,
			false);
  const arma::icube path_action(path_action_.begin(), n_dec, n_pos, n_path, false);
  // Computing the primal and dual values
  arma::cube primal(n_dec, n_pos, n_path);
  arma::cube dual(n_dec, n_pos, n_path);
  // Initialise with the last decision epoch
  int t = n_dec - 1;
  std::size_t a, i, p;
  arma::mat state(n_path, n_dim);
  arma::mat reward(n_path, n_action * n_pos);
  state = path(arma::span(t), arma::span::all, arma::span::all);
  reward = Rcpp::as<arma::mat>(Reward_(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), t + 1));
  for (p = 0; p < n_pos; p++) {
    dual.tube(t, p) =
        arma::max(reward.cols(p * n_action, (p + 1) * n_action - 1), 1);
  }
  primal.tube(arma::span(t), arma::span::all) =
      dual.tube(arma::span(t), arma::span::all);
  // Perform the backward induction
  arma::uword policy;
  if (full_control) {  // For the full control case
    arma::uword next;
    for (t = (n_dec - 2); t >= 0; t--) {
      state = path(arma::span(t), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::mat>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), t + 1));
#pragma omp parallel for private(p, i, policy, next, a)
      for (p = 0; p < n_pos; p++) {
        for (i = 0; i < n_path; i++) {
          // Primal values
          policy = path_action(t, p, i) - 1;
          next = control(p, policy) - 1;
          primal(t, p, i) = reward(i, p * n_action + policy) + mart(t, next, i)
              + primal(t + 1, next, i);
          // Dual values
          next = control(p, 0) - 1;
          dual(t, p, i) = reward(i, p * n_action) + mart(t, next, i)
              + dual(t + 1, next, i);
          for (a = 1; a < n_action; a++) {
            next = control(p, a) - 1;
            dual(t, p, i) =
                std::max(reward(i, p * n_action + a) + mart(t, next, i)
                         + dual(t + 1, next, i), dual(t, p, i));
          }
        }
      }
    }
  } else {  // Positions evolve randomly
    std::size_t k;
    double expected;
    arma::rowvec prob_weight(n_pos);
    for (t = (n_dec - 2); t >= 0; t--) {
      state = path(arma::span(t), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::mat>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), t + 1));
#pragma omp parallel for private(p, i, policy, prob_weight, expected, a)
      for (p = 0; p < n_pos; p++) {
        for (i = 0; i < n_path; i++) {
          //  Primal values
          policy = path_action(t, p, i) - 1;
          prob_weight = control2.tube(p, policy);
          primal(t, p, i) = reward(i, p * n_action + policy) +
              mart(t, policy, p * n_path + i) +
              arma::sum(primal.slice(i).row(t + 1) % prob_weight);
          // Dual values
          prob_weight = control2.tube(p, 0);
          expected = arma::sum(dual.slice(i).row(t + 1) % prob_weight);
          dual(t, p, i) = reward(i, p * n_action) + mart(t, 0, p * n_path + i)
              + expected;
          for (a = 1; a < n_action; a++) {
            prob_weight = control2.tube(p, a);
            expected = arma::sum(dual.slice(i).row(t + 1) % prob_weight);
            dual(t, p, i) = std::max(reward(i, p * n_action + a) +
                                     mart(t, a, p * n_path + i) + expected,
                                     dual(t, p, i));
          }
        }
      }
    }
  }
  return Rcpp::List::create(Rcpp::Named("primal") = primal,
                            Rcpp::Named("dual") = dual);
}

// Export to R
RcppExport SEXP rcss_Duality(SEXP pathSEXP,
                             SEXP controlSEXP,
                             SEXP RewardSEXP,
                             SEXP martSEXP,
                             SEXP path_actionSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        path(pathSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        control(controlSEXP);
    Rcpp::traits::input_parameter<Rcpp::Function>::type
        Reward(RewardSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        mart(martSEXP);
    Rcpp::traits::input_parameter<Rcpp::IntegerVector>::type
        path_action(path_actionSEXP);
    __result = Rcpp::wrap(Duality(path, control, Reward, mart,
                                  path_action));
    return __result;
END_RCPP
}
