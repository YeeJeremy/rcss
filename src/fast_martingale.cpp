// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Fast martingale increments
////////////////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
#include <omp.h>
#endif
#include <RcppArmadillo.h>
#include <Rcpp.h>

arma::cube FastMartingale(Rcpp::NumericVector value_,
                          Rcpp::NumericVector disturb_,
                          Rcpp::NumericVector weight_,
                          Rcpp::NumericVector path_,
                          Rcpp::IntegerVector path_nn_,
                          Rcpp::Function Neighbour_,
                          Rcpp::NumericMatrix grid_,
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
  const arma::vec weight(weight_.begin(), weight_.length(), false);
  const arma::ivec p_dims = path_.attr("dim");
  const std::size_t n_path = p_dims(1);
  const arma::mat path(path_.begin(), n_dec * n_path, n_dim, false);
  const arma::imat path_nn(path_nn_.begin(), n_dec, n_path, false);
  const arma::mat grid(grid_.begin(), n_grid, n_dim, false);
  const arma::ivec c_dims = control_.attr("dim");
  arma::cube control;  // control is initialised if position evolution random
  bool full_control = true;
  std::size_t n_action = c_dims(1);
  if (c_dims.n_elem == 3) {  // If position evolution is random
    full_control = false;
    arma::cube temp_control(control_.begin(), n_pos, n_action, n_pos, false);
    control = temp_control;
  }
  // Initialise the martingales with the expected values
  arma::cube mart(n_dec - 1, n_pos, n_path);
  std::size_t i, j, k, l;
  arma::mat state(n_path * n_subsim, n_dim);
  Rcpp::IntegerVector r_host(n_path * n_subsim);
  arma::uvec host(n_path * n_subsim);
  arma::mat temp_value(n_grid, n_dim);
  arma::mat temp_disturb(n_dim, n_dim);
  Rcpp::Rcout << "Subsimulation at dec: ";
  for (i = 0; i < (n_dec - 1); i++) {
    Rcpp::Rcout << i << "...";
#pragma omp parallel for private(j, k, l)
    for (j = 0; j < n_subsim; j++) {
      for (k = 0; k < n_path; k++) {
        l = i * (n_subsim * n_path) + k * n_subsim + j;
        state.row(j * n_path + k) = path.row(k * n_dec + i) *
            arma::trans(disturb.cols(l * n_dim, (l + 1) * n_dim - 1));
      }
    }
    r_host = 
      Neighbour_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)),
		 Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid)));
    host = Rcpp::as<arma::uvec>(r_host);
    host -= 1;  // R indexing to C++
#pragma omp parallel for private(j, l, temp_value)
    for (j = 0; j < n_pos; j++) {
      l = (i + 1) * n_dim * n_pos;
      temp_value = value.cols(l + j * n_dim, l + (j + 1) * n_dim - 1);
      mart.tube(i, j) = arma::reshape(
          arma::sum(state % temp_value.rows(host), 1), n_path, n_subsim)
          * weight;
    }
  }
  // Subtracting the path realizations
  arma::rowvec temp(n_pos * n_dim);
  arma::uvec seq1 = arma::linspace<arma::uvec>(2 * n_dim - 1, n_pos * n_dim - 1,
                                              n_pos - 1);
  arma::uvec seq2 = seq1 - n_dim;
  Rcpp::Rcout << "Subtracting path realisations...";
  for (i = 0; i < n_path; i++) {
    for (j = 0; j < (n_dec - 1); j++) {
      temp = arma::repmat(path.row(i * n_dec + j + 1), 1, n_pos);
      temp = arma::cumsum(value.cols((j + 1) * n_dim * n_pos,
                                     (j + 2) * n_dim * n_pos - 1)
                          .row(path_nn(j + 1, i) - 1) % temp, 1);
      mart(j, 0, i) -= temp(n_dim - 1);
      mart(arma::span(j), arma::span(1, n_pos - 1), arma::span(i))
          -= temp(seq1) - temp(seq2);
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
RcppExport SEXP rcss_FastMartingale(SEXP valueSEXP,
                                    SEXP subsim_disturbSEXP,
                                    SEXP subsim_weightSEXP,
                                    SEXP pathSEXP,
                                    SEXP path_neighbourSEXP,
                                    SEXP NeighbourSEXP,
                                    SEXP gridSEXP,
                                    SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        value(valueSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        subsim_disturb(subsim_disturbSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        subsim_weight(subsim_weightSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        path(pathSEXP);
    Rcpp::traits::input_parameter<Rcpp::IntegerVector>::type
        path_neighbour(path_neighbourSEXP);
    Rcpp::traits::input_parameter<Rcpp::Function>::type
        Neighbour(NeighbourSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericMatrix>::type
        grid(gridSEXP);
    Rcpp::traits::input_parameter<Rcpp::NumericVector>::type
        control(controlSEXP);
    __result =
        Rcpp::wrap(FastMartingale(value, subsim_disturb, subsim_weight, path,
                                  path_neighbour, Neighbour, grid, control));
    return __result;
END_RCPP
}
