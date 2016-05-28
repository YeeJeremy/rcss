// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Performs the bellman recursion using row rearrangement
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/bellman.h"
#include "inst/include/slow.h"
#include <Rcpp.h>

// Perform bellman recursion using row rearrangement
Rcpp::List Bellman(Rcpp::NumericMatrix grid_,
                   Rcpp::NumericVector reward_,
                   Rcpp::NumericVector control_,
                   Rcpp::NumericVector disturb_,
                   Rcpp::NumericVector weight) {
  // Passing R objects to C++
  const std::size_t n_grid = grid_.nrow();
  const std::size_t n_dim = grid_.ncol();
  const arma::mat grid(grid_.begin(), n_grid, n_dim, false);
  const arma::ivec r_dims = reward_.attr("dim");
  const std::size_t n_pos = r_dims(2);
  const std::size_t n_action = r_dims(3);
  const std::size_t n_dec = r_dims(4);
  const arma::mat reward(reward_.begin(), n_grid, n_dim * n_pos * n_action * n_dec,
			 false);
  const arma::ivec c_dims = control_.attr("dim");
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
  const arma::ivec d_dims = disturb_.attr("dim");
  const std::size_t n_disturb = d_dims(2);
  const arma::mat disturb(disturb_.begin(), n_dim, n_dim * n_disturb, false);
  // Bellman recursion
  arma::cube value(n_grid, n_dim * n_pos, n_dec);
  arma::ucube action(n_grid, n_pos, n_dec);
  arma::cube cont(n_grid, n_dim * n_pos, n_dec, arma::fill::zeros);
  arma::mat d_value(n_grid, n_dim);
  Rcpp::Rcout << "At dec: " << n_dec - 1 << "...";
  if (full_control) {
    BellmanOptimal(grid, control, &value, reward, &cont, &action, n_dec - 1);
  } else {
    BellmanOptimal2(grid, control2, &value, reward, &cont, &action, n_dec - 1);
  }
  for (int i = (n_dec - 2); i >= 0; i--) {
    Rcpp::Rcout << i;
    // Approximating the continuation value
    for (std::size_t j = 0; j < n_disturb; j++) {
      for (std::size_t k = 0; k < n_pos; k++) {
        d_value = value.slice(i + 1).cols(k * n_dim, (k + 1) * n_dim - 1)
            * disturb.cols(j * n_dim, (j + 1) * n_dim - 1);
        cont.slice(i).cols(k * n_dim, (k + 1) * n_dim - 1) +=
            weight[j] * Optimal(grid, d_value);
      }
    }
    Rcpp::Rcout << "..";
    // Find optimal action and the corresponding value function
    if (full_control) {
      BellmanOptimal(grid, control, &value, reward, &cont, &action, i);
    } else {
      BellmanOptimal2(grid, control2, &value, reward, &cont, &action, i);
    }
    Rcpp::Rcout << ".";
  }
  action += 1;  // C++ indexing to R
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("expected") = cont,
                            Rcpp::Named("action") = action);
}

// Export to R
RcppExport SEXP rcss_Bellman(SEXP gridSEXP,
                             SEXP rewardSEXP,
                             SEXP controlSEXP,
                             SEXP disturbSEXP,
                             SEXP weightSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericMatrix>::type
        grid(gridSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        reward(rewardSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        control(controlSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        disturb(disturbSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        weight(weightSEXP);
    __result = Rcpp::wrap(Bellman(grid, reward, control, disturb, weight));
    return __result;
END_RCPP
}
