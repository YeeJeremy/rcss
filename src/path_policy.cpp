// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Returns the prescribed policy for supplied paths
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>

arma::ucube PathPolicy(Rcpp::NumericVector path_,
                       Rcpp::IntegerVector path_nn_,
                       Rcpp::NumericVector control_,
                       Rcpp::Function Reward_,
                       Rcpp::NumericVector expected_) {
  // R objects into C++
  const arma::ivec p_dims = path_.attr("dim");
  const int n_dec = p_dims(0);
  const std::size_t n_path = p_dims(1);
  const std::size_t n_dim = p_dims(2);
  const arma::cube path(path_.begin(), n_dec, n_path, n_dim, false);
  const arma::imat path_nn(path_nn_.begin(), n_dec, n_path, false);
  const arma::ivec c_dims = control_.attr("dim");
  const std::size_t n_pos = c_dims(0);
  const std::size_t n_action = c_dims(1);
  arma::imat control;
  arma::cube control2;
  bool full_control;
  if (c_dims.n_elem == 3) {  // Partial control
    full_control = false;
    arma::cube temp_control2(control_.begin(), n_pos, n_action, n_pos, false);
    control2 = temp_control2;
  } else {  // Full control
    full_control = true;
    arma::mat temp_control(control_.begin(), n_pos, n_action, false);
    control = arma::conv_to<arma::imat>::from(temp_control);
  }
  const arma::ivec e_dims = expected_.attr("dim");
  const std::size_t n_grid = e_dims(0);
  const arma::cube cont(expected_.begin(), n_grid, n_dim * n_pos, n_dec, false);
  // Finding the optimal action
  arma::ucube policy(n_dec, n_pos, n_path);
  arma::mat state(n_path, n_dim);
  arma::mat reward(n_path, n_action * n_pos);
  arma::mat temp_value1(n_grid, n_dim);
  arma::mat temp_value2(n_path, n_dim);
  arma::mat compare_value(n_path, n_action);
  arma::uvec host(n_path);
  arma::uword best;
  int t;
  std::size_t p, a, n, i;
  if (full_control) {  // Full control
    for (t = 0; t < n_dec; t++) {
      state = path(arma::span(t), arma::span::all, arma::span::all);
      host = arma::conv_to<arma::uvec>::from(path_nn.row(t)) - 1;
      reward = Rcpp::as<arma::mat>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), t + 1));
      for (p = 0; p < n_pos; p++) {
        for (a = 0; a < n_action; a++) {
          n = control(p, a) - 1;
          temp_value1 = cont.slice(t).cols(n * n_dim, (n + 1) * n_dim - 1);
          temp_value2 = temp_value1.rows(host);
          compare_value.col(a) = reward.col(n_action * p + a) +
              arma::sum(temp_value2 % state, 1);
        }
        for (i = 0; i < n_path; i++) {
          compare_value.row(i).max(best);
          policy(t, p, i) = best;
        }
      }
    }
  } else {  // Evolution of position is random
    for (t = 0; t < n_dec; t++) {
      state = path(arma::span(t), arma::span::all, arma::span::all);
      host = arma::conv_to<arma::uvec>::from(path_nn.row(t)) - 1;
      reward = Rcpp::as<arma::mat>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), t + 1));
      for (p = 0; p < n_pos; p++) {
        for (a = 0; a < n_action; a++) {
          compare_value.col(a) = reward.col(n_action * p + a);
          for (i = 0; i < n_pos; i++) {
            temp_value1 = cont.slice(t).cols(i * n_dim, (i + 1) * n_dim - 1);
            temp_value2 = temp_value1.rows(host);
            compare_value.col(a) += control2(p, a, i) *
                arma::sum(temp_value2 % state, 1);
          }
        }
        for (i = 0; i < n_path; i++) {
          compare_value.row(i).max(best);
          policy(t, p, i) = best;
        }
      }
    }
  }
  return (policy + 1);
}

// Export to R
RcppExport SEXP rcss_PathPolicy(SEXP pathSEXP,
                                SEXP path_nnSEXP,
                                SEXP controlSEXP,
                                SEXP RewardSEXP,
                                SEXP expectedSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        path(pathSEXP);
    Rcpp::traits::input_parameter<Rcpp::IntegerVector>::type
        path_nn(path_nnSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        control(controlSEXP);
    Rcpp::traits::input_parameter<Rcpp::Function>::type
        Reward(RewardSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        expected(expectedSEXP);
    __result = Rcpp::wrap(PathPolicy(path, path_nn, control, Reward, expected));
    return __result;
END_RCPP
}
