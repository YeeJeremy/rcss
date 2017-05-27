// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Incorrect martingale increments for finite distribution
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>

//[[Rcpp::export]]
arma::cube WrongMartingale(Rcpp::NumericVector value_,
                           Rcpp::NumericVector expected_,
                           Rcpp::NumericVector path_,
                           Rcpp::NumericVector control_) {
  // R objects to C++
  const arma::ivec v_dims = value_.attr("dim");
  const std::size_t n_grid = v_dims(0);
  const std::size_t n_dim = v_dims(1);
  const std::size_t n_pos = v_dims(2);
  const std::size_t n_dec = v_dims(3);
  const arma::mat value(value_.begin(), n_grid, n_dim * n_pos * n_dec, false);
  const arma::mat expected(expected_.begin(), n_grid, n_dim * n_pos * n_dec, false);
  const arma::ivec p_dims = path_.attr("dim");
  const std::size_t n_path = p_dims(1);
  const arma::cube path(path_.begin(), n_dec, n_path, n_dim, false);
  const arma::ivec c_dims = control_.attr("dim");
  arma::cube control;  // control is initialised if position evolution random
  bool full_control = true;
  std::size_t n_action = c_dims(1);
  if (c_dims.n_elem == 3) {  // If position evolution is random
    full_control = false;
    arma::cube temp_control(control_.begin(), n_pos, n_action, n_pos, false);
    control = temp_control;
  }  
  // Martingale increments
  arma::cube mart(n_dec - 1, n_pos, n_path);
  arma::mat state(n_path, n_dim);
  arma::mat state_t(n_dim, n_path);
  arma::mat next(n_path, n_dim);
  arma::mat next_t(n_dim, n_path);
  arma::mat temp_value(n_grid, n_dim);
  arma::mat temp_expected(n_grid, n_dim);
  std::size_t i, j, k, l, m;
  Rcpp::Rcout << "At dec: ";
  for (i = 0; i < (n_dec -1); i++) {
    Rcpp::Rcout << i << "...";
    state = path.tube(arma::span(i), arma::span::all);
    state_t = state.t();
    next = path.tube(arma::span(i + 1), arma::span::all);
    next_t = next.t();
    for (j = 0; j < n_pos; j++) {
      l = i * n_dim * n_pos;
      m = (i + 1) * n_dim * n_pos;
      temp_value = value.cols(m + j * n_dim, m + (j + 1) * n_dim - 1);
      temp_expected = expected.cols(l + j * n_dim, l + (j + 1) * n_dim - 1);
      // Finding the average using expected value function
      mart.tube(i, j) = arma::max(temp_expected * state_t, 0);
      // Subtracting the path realization
      mart.tube(i, j) -= arma::max(temp_value * next_t, 0);
    }
  }
  if (full_control) {  // If full control, then done
    Rcpp::Rcout << "Done.\n";
    return mart;
  } else {  // Random position evolution
    arma::cube mart2(n_dec - 1, n_action, n_pos * n_path);
    arma::mat temp_mart(n_pos, n_path);
    arma::mat prob_weight(1, n_pos);
    for (i = 0; i < (n_dec - 1); i++) {
      temp_mart = mart(arma::span(i), arma::span::all, arma::span::all);
      for (j = 0; j < n_pos; j++) {
	for (k = 0; k < n_action; k++) {
	  prob_weight = control.tube(j, k);
	  mart2(arma::span(i), arma::span(k),
		arma::span(j * n_path, (j + 1) * n_path - 1)) =
              prob_weight * temp_mart;
	}
      }
    }
    Rcpp::Rcout << "Done.\n";
    return mart2;
  }
}
