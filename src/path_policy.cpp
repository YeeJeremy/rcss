// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Prescribed policy for supplied paths
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/policy.h"

// Extract the prescribed policy using row rearrangment
//[[Rcpp::export]]
arma::ucube PathPolicy(const arma::cube& path,
                       Rcpp::NumericVector control_,
                       Rcpp::Function Reward_,
                       Rcpp::NumericVector expected_) {
  // R objects into C++
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
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
  const arma::cube cont(expected_.begin(), n_grid, n_dim * n_pos, n_dec - 1, false);
  // Finding the optimal action
  arma::ucube policy(n_path, n_pos, n_dec - 1);
  if (full_control) {
    SlowOptimalPolicy(policy, path, control, Reward_, cont, n_dec,
                      n_path, n_dim, n_pos, n_action);
  } else {
    SlowOptimalPolicy2(policy, path, control2, Reward_, cont, n_dec,
                       n_path, n_dim, n_pos, n_action);
  }
  return (policy + 1);
}

// Extract the prescribed policy using nearest neighbours
//[[Rcpp::export]]
arma::ucube FastPathPolicy(const arma::cube& path,
                           const arma::mat& grid,
                           Rcpp::NumericVector control_,
                           Rcpp::Function Reward_,
                           Rcpp::NumericVector expected_) {
  // R objects into C++
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
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
  const arma::cube cont(expected_.begin(), n_grid, n_dim * n_pos, n_dec - 1, false);
  // Finding the optimal action
  arma::ucube policy(n_path, n_pos, n_dec - 1);
  if (full_control) {
    FastOptimalPolicy(policy, path, grid, control, Reward_, cont, n_dec,
                      n_path, n_dim, n_pos, n_action);
  } else {
    FastOptimalPolicy2(policy, path, grid, control2, Reward_, cont, n_dec,
                       n_path, n_dim, n_pos, n_action);
  }
  return (policy + 1);
}
