// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Additive duals for finite distribution
////////////////////////////////////////////////////////////////////////////////

#include <rflann.h>
#include <string>
#include <RcppArmadillo.h>

// Maximum value using nearest neighbours + row-rearrange
arma::vec OptimalValue(const arma::mat& state,
                       const arma::mat& subgradient,
                       const arma::umat& neighbour) {
  const std::size_t n_state = state.n_rows;
  const std::size_t n_dim = state.n_cols;
  const std::size_t n_neighbour = neighbour.n_cols;
  arma::mat compare(n_state, n_neighbour);
  arma::uvec cols_ind = arma::linspace<arma::uvec>(0, n_dim - 1, n_dim);
  for (std::size_t nn = 0; nn < n_neighbour; nn++) {
    compare.col(nn) = arma::sum(subgradient.submat(neighbour.col(nn), cols_ind)
                                % state, 1);
  }
  arma::vec max_value(n_state);
  max_value = arma::max(compare, 1);
  return max_value;
}

// Additive duals using slow, accelerated and fast methods
//[[Rcpp::export]]
arma::cube FiniteAddDual(const arma::cube& path,
                         Rcpp::NumericVector path_disturb_,
                         const arma::mat& grid,
                         Rcpp::NumericVector value_,
                         Rcpp::NumericVector expected_,
                         const std::string& build,
                         const std::size_t& k) {
  // R objects to C++
  const arma::ivec d_dims = path_disturb_.attr("dim");
  const std::size_t n_dim = d_dims(0);
  const std::size_t n_path = d_dims(2);
  const std::size_t n_dec = d_dims(3) + 1;
  const arma::cube path_disturb(path_disturb_.begin(), n_dim, n_dim * n_path,
                                n_dec - 1, false);
  const arma::ivec v_dims = value_.attr("dim");
  const std::size_t n_grid = v_dims(0);
  const std::size_t n_pos = v_dims(2);
  const arma::cube value(value_.begin(), n_grid, n_dim * n_pos, n_dec, false);
  const arma::cube expected(expected_.begin(), n_grid, n_dim * n_pos,
                            n_dec - 1, false);
  // Computing the path neighbours
  arma::uvec path_nn(n_path * (n_dec - 1));
  {
    arma::mat path_nodes(n_path * (n_dec - 1), n_dim);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      path_nodes.rows(n_path * tt, n_path * (tt + 1) - 1) = path.slice(tt);
    }
    path_nn = arma::conv_to<arma::uvec>::from(rflann::FastKDNeighbour(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path_nodes.cols(1, n_dim - 1))),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid.cols(1, n_dim - 1))),
        1)) - 1;
  }
  // Compute the additive duals
  arma::cube mart(n_path, n_pos, n_dec - 1);
  arma::mat state(n_path, n_dim);
  arma::mat dstate(n_path, n_dim);
  arma::mat cont(n_grid, n_dim);
  arma::uvec host(n_path);
  if (build == "fast") {
    arma::mat fitted(n_grid, n_dim);
    arma::uvec host2(n_path);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      Rcpp::Rcout << tt << "...";
      host = path_nn.subvec(n_path * tt, n_path * (tt + 1) - 1);
      state = grid.rows(host);  // Finding the nearest neighbour states
      // Disturbing the nearest neighbour states
      for (std::size_t pp = 0; pp < n_path; pp++) {
        dstate.row(pp) = state.row(pp) *
            arma::trans(path_disturb.slice(tt).cols(n_dim * pp, n_dim * (pp + 1) - 1));
      }
      host2 = arma::conv_to<arma::uvec>::from(rflann::FastKDNeighbour(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(dstate.cols(1, n_dim - 1))),
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid.cols(1, n_dim - 1))),
          1)) - 1;
      // Calculating the martingale increments
      for (std::size_t pos = 0; pos < n_pos; pos++) {
        cont = expected.slice(tt).cols(n_dim * pos, n_dim * (pos + 1) - 1);
        fitted = value.slice(tt + 1).cols(n_dim * pos, n_dim * (pos + 1) - 1);
        mart.slice(tt).col(pos) = arma::sum(cont.rows(host) % state, 1) -
            arma::sum(fitted.rows(host2) % dstate, 1);
      }
    }
  } else if (build == "accelerated") {
    arma::mat fitted(n_grid, n_dim);
    arma::umat host2(n_path, k);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      Rcpp::Rcout << tt << "...";
      host = path_nn.subvec(n_path * tt, n_path * (tt + 1) - 1);
      state = grid.rows(host);  // Finding the nearest neighbour states
      // Disturbing the nearest neighbour states
      for (std::size_t pp = 0; pp < n_path; pp++) {
        dstate.row(pp) = state.row(pp) *
            arma::trans(path_disturb.slice(tt).cols(n_dim * pp, n_dim * (pp + 1) - 1));
      }
      host2 = arma::conv_to<arma::umat>::from(rflann::FastKDNeighbour(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(dstate.cols(1, n_dim - 1))),
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid.cols(1, n_dim - 1))),
          k)) - 1;
      // Calculating the martingale increments
      for (std::size_t pos = 0; pos < n_pos; pos++) {
        cont = expected.slice(tt).cols(n_dim * pos, n_dim * (pos + 1) - 1);
        fitted = value.slice(tt + 1).cols(n_dim * pos, n_dim * (pos + 1) - 1);
        mart.slice(tt).col(pos) = arma::sum(cont.rows(host) % state, 1) -
            OptimalValue(dstate, fitted, host2);
      }
    }
  } else if (build == "slow") {
    arma::mat fitted(n_dim, n_grid);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      Rcpp::Rcout << tt << "...";
      host = path_nn.subvec(n_path * tt, n_path * (tt + 1) - 1);
      state = grid.rows(host);  // Finding the nearest neighbour states
      // Disturbing the nearest neighbour states
      for (std::size_t pp = 0; pp < n_path; pp++) {
        dstate.row(pp) = state.row(pp) *
            arma::trans(path_disturb.slice(tt).cols(n_dim * pp, n_dim * (pp + 1) - 1));
      }
      // Calculating the martingale increments
      for (std::size_t pos = 0; pos < n_pos; pos++) {
        cont = expected.slice(tt).cols(n_dim * pos, n_dim * (pos + 1) - 1);
        fitted =
            arma::trans(value.slice(tt + 1).cols(n_dim * pos, n_dim * (pos + 1) - 1));
        mart.slice(tt).col(pos) = arma::sum(cont.rows(host) % state, 1) -
            arma::max(dstate * fitted, 1);
      }
    }
  }
  Rcpp::Rcout << "Done.\n";
  return mart;
}
