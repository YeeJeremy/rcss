// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Computing the martingale increments using the row rearrangement operator
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <RcppArmadillo.h>
#include <Rcpp.h>

// Finds the maximum value
arma::vec OptimalValue(const arma::mat& grid, const arma::mat& subgradient) {
  const arma::mat t_grid = grid.t();
  const std::size_t n_grid = grid.n_rows;
  arma::vec optimal(n_grid);
  std::size_t i;
#pragma omp parallel for private(i)
  for (i = 0; i < n_grid; i++) {
    optimal(i) = (subgradient * t_grid.col(i)).max();
  }
  return optimal;
}

// Calculate the martingale increments using the row rearrangement
arma::cube Martingale(Rcpp::NumericVector value_,
                      Rcpp::NumericVector disturb_,
                      Rcpp::NumericVector weight,
                      Rcpp::NumericVector path_,
                      Rcpp::NumericVector control_) {
  // R objects to C++
  const arma::ivec v_dims = value_.attr("dim");
  const std::size_t n_grid = v_dims(0);
  const std::size_t n_dim = v_dims(1);
  const std::size_t n_pos = v_dims(2);
  const std::size_t n_dec = v_dims(3);
  const arma::mat value(value_.begin(), n_grid, n_dim * n_pos * n_dec, false);
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_subsim = d_dims(2);
  const std::size_t n_sim = n_subsim * d_dims(3) * d_dims(4);
  const arma::mat disturb(disturb_.begin(), n_dim, n_dim * n_sim, false);
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
  arma::mat next(n_path, n_dim);
  arma::mat temp_state(n_path * n_subsim, n_dim);
  arma::mat temp_store(n_path, n_subsim);
  arma::mat temp_value(n_grid, n_dim);
  std::size_t i, j, k, l, m;
  Rcpp::Rcout << "At dec: ";
  for (i = 0; i < (n_dec -1); i++) {
    Rcpp::Rcout << i << "...";
    state = path.tube(arma::span(i), arma::span::all);
    next = path.tube(arma::span(i + 1), arma::span::all);
    for (j = 0; j < n_pos; j++) {
      l = (i + 1) * n_dim * n_pos;
      temp_value = value.cols(l + j * n_dim, l + (j + 1) * n_dim - 1);
      // Finding the average
#pragma omp parallel for private(k, m, l)
      for (k = 0; k < n_subsim; k++) {
        for (m = 0; m < n_path; m++) {
          l = i * (n_subsim * n_path) + m * n_subsim + k;
          temp_state.row(k * n_path + m) = weight[k] * state.row(m)
              * arma::trans(disturb.cols(l * n_dim, (l + 1) * n_dim - 1));
        }
      }
      temp_store =
          reshape(OptimalValue(temp_state, temp_value), n_path, n_subsim);
      mart.tube(i, j) = arma::sum(temp_store, 1);
      // Subtracting the path realization
      mart.tube(i, j) -= OptimalValue(next, temp_value);
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

// Export to R
RcppExport SEXP rcss_Martingale(SEXP valueSEXP,
                                SEXP disturbSEXP,
                                SEXP weightSEXP,
                                SEXP pathSEXP,
                                SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        value(valueSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        disturb(disturbSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        weight(weightSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        path(pathSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        control(controlSEXP);
    __result = Rcpp::wrap(Martingale(value, disturb, weight, path, control));
    return __result;
END_RCPP
}
