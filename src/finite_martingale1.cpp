// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Martingale increments for finite distribution using row-rearrangement
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>

//[[Rcpp::export]]
arma::cube FiniteMartingale1(Rcpp::NumericMatrix grid_,
                             Rcpp::NumericVector value_,
                             Rcpp::NumericVector expected_,
                             Rcpp::NumericVector path_disturb_,
                             Rcpp::IntegerVector path_nn_,
                             Rcpp::NumericVector control_) {
  // R objects to C++
  const arma::ivec g_dims = grid_.attr("dim");
  const std::size_t n_grid = g_dims(0);
  const std::size_t n_dim = g_dims(1);
  const arma::mat grid(grid_.begin(), n_grid, n_dim, false);
  const arma::ivec v_dims = value_.attr("dim");
  const std::size_t n_pos = v_dims(2);
  const std::size_t n_dec = v_dims(3);
  const arma::mat value(value_.begin(), n_grid, n_dim * n_pos * n_dec, false);
  const arma::mat expected(expected_.begin(), n_grid, n_dim * n_pos * n_dec, false);
  const arma::ivec p_dims = path_disturb_.attr("dim");
  const std::size_t n_path = p_dims(3);
  const arma::mat path_disturb(path_disturb_.begin(), n_dim, n_dim * (n_dec - 1) *
			 n_path, false);
  const arma::imat path_nn(path_nn_.begin(), n_dec, n_path, false);
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
  arma::mat disturb_state(n_path, n_dim);
  arma::mat disturb_state_t(n_dim, n_path);
  arma::mat temp_value(n_grid, n_dim);
  arma::mat temp_expected(n_grid, n_dim);
  const arma::umat path_nn2 = arma::trans(arma::conv_to<arma::umat>::from(path_nn));
  arma::uvec host(n_path);
  std::size_t i, j, k, l, m, n;
  Rcpp::Rcout << "At dec: ";
  for (i = 0; i < (n_dec -1); i++) {
    Rcpp::Rcout << i << "...";
    host = path_nn2.col(i) - 1;
    state = grid.rows(host);  // Finding the nearest neighbours for points
    // Disturbing the grid
    for (k = 0; k < n_path; k++) {
      n = k * (n_dec - 1) + i;
      disturb_state.row(k) = state.row(k) * 
	arma::trans(path_disturb.cols(n * n_dim, (n + 1) * n_dim - 1));
    }
    disturb_state_t = disturb_state.t();
    // Calculating the martingale increments
    l = i * n_dim * n_pos;  // start index for expected array
    m = (i + 1) * n_dim * n_pos;  // start index for value array
    for (j = 0; j < n_pos; j++) {
      temp_expected = expected.cols(l + j * n_dim, l + (j + 1) * n_dim - 1);
      temp_value = value.cols(m + j * n_dim, m + (j + 1) * n_dim - 1);
      mart.tube(i, j) = arma::sum(temp_expected.rows(host) % state, 1) -
          arma::trans(arma::max(temp_value * disturb_state_t, 0));
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
