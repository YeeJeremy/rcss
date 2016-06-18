// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Testing the prescibed policy using generated sample paths
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>

arma::uword NextPosition(const arma::vec &prob_weight) {
  const std::size_t n_pos = prob_weight.n_elem;
  const arma::vec cum_prob = arma::cumsum(prob_weight);
  const double rand_unif = R::runif(0, 1);
  arma::uword next_state = 0;
  for (std::size_t i = 1; i < n_pos; i++) {
    if (rand_unif <= cum_prob(i)) {
      next_state = i;
      break;
    }
  }
  return next_state;
}

arma::vec TestPolicy(int start_position,
                     Rcpp::NumericVector path_,
                     Rcpp::NumericVector control_,
                     Rcpp::Function Reward_,
                     Rcpp::IntegerVector path_action_) {
  // R objects into C++
  const arma::ivec p_dims = path_.attr("dim");
  const int n_dec = p_dims(0);
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
  const arma::icube path_action(path_action_.begin(), n_dec, n_pos, n_path,
				false);
  // Performing out of sample testing of the prescribed policy
  arma::vec value(n_path, arma::fill::zeros);
  arma::uvec pos(n_path);
  pos.fill(start_position - 1);  // Initialise with starting position
  arma::mat state(n_path, n_dim);
  arma::mat reward(n_path, n_action * n_pos);
  arma::uword policy;
  if (full_control) {
    for (int t = 0; t < n_dec; t++) {
      state = path(arma::span(t), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::mat>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), t + 1));
      for (std::size_t i = 0; i < n_path; i++) {
        policy = path_action(t, pos(i), i) - 1;
        value(i) += reward(i, pos(i) * n_action + policy);
        pos(i) = control(pos(i), policy) - 1;
      }
    }
  } else {
    arma::vec prob_weight(n_pos);
    for (int t = 0; t < n_dec; t++) {
      state = path(arma::span(t), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::mat>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), t + 1));
      for (std::size_t i = 0; i < n_path; i++) {
        policy = path_action(t, pos(i), i) - 1;
        value(i) += reward(i, pos(i) * n_action + policy);
        prob_weight = control2.tube(pos(i), policy);
        pos(i) = NextPosition(prob_weight);
      }
    }
  }
  return value;
}

// Export to R
RcppExport SEXP rcss_TestPolicy(SEXP start_positionSEXP,
                                SEXP pathSEXP,
                                SEXP controlSEXP,
                                SEXP RewardSEXP,
                                SEXP path_actionSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<int>::type
        start_position(start_positionSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        path(pathSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        control(controlSEXP);
    Rcpp::traits::input_parameter<Rcpp::Function>::type
        Reward(RewardSEXP);
    Rcpp::traits::input_parameter<Rcpp::IntegerVector >::type
        path_action(path_actionSEXP);
    __result = Rcpp::wrap(TestPolicy(start_position, path, control,
                                     Reward, path_action));
    return __result;
END_RCPP
}
