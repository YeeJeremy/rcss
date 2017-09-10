// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Prescribed policy for supplied paths using various methods
////////////////////////////////////////////////////////////////////////////////

#include <rflann.h>
#include "inst/include/slow.h"
#include "inst/include/policy.h"

// Full control and compare all subgradients
void SlowOptimalPolicy(arma::ucube& policy,
                       const arma::cube& path,
                       const arma::imat& control,
                       Rcpp::Function Reward_,
                       const arma::cube& cont,
                       const std::size_t& n_dec,
                       const std::size_t& n_path,
                       const std::size_t& n_dim,
                       const std::size_t& n_pos,
                       const std::size_t& n_action) {
  arma::cube reward(n_path, n_action, n_pos);
  arma::mat compare(n_path, n_action);
  std::size_t nn;
  arma::mat fitted(n_path, n_dim);
  for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
    reward = Rcpp::as<arma::cube>(Reward_(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t aa = 0; aa < n_action; aa++) {
        nn = control(pp, aa) - 1;
        fitted = Optimal(path.slice(tt),
                         cont.slice(tt).cols(n_dim * nn, n_dim * (nn + 1) - 1));
        compare.col(aa) = reward.slice(pp).col(aa) +
            arma::sum(fitted % path.slice(tt), 1);
      }
      policy.slice(tt).col(pp) = arma::index_max(compare, 1);
    }
  }
}

// Partial control and compare all subgradients
void SlowOptimalPolicy2(arma::ucube& policy,
                        const arma::cube& path,
                        const arma::cube& control2,
                        Rcpp::Function Reward_,
                        const arma::cube& cont,
                        const std::size_t& n_dec,
                        const std::size_t& n_path,
                        const std::size_t& n_dim,
                        const std::size_t& n_pos,
                        const std::size_t& n_action) {
  arma::cube reward(n_path, n_action, n_pos);
  arma::mat compare(n_path, n_action);
  arma::mat fitted(n_path, n_dim);
  for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
    reward = Rcpp::as<arma::cube>(Reward_(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t aa = 0; aa < n_action; aa++) {
        compare.col(aa) = reward.slice(pp).col(aa);
        for (std::size_t pos = 0; pos < n_pos; pos++) {
          fitted = Optimal(path.slice(tt),
                           cont.slice(tt).cols(n_dim * pos, n_dim * (pos + 1) - 1));
          compare.col(aa) +=
              control2(pp, aa, pos) * arma::sum(fitted % path.slice(tt), 1);
        }
      }
      policy.slice(tt).col(pp) = arma::index_max(compare, 1);
    }
  }
}

// Full control and nearest neighbour
void FastOptimalPolicy(arma::ucube& policy,
                       const arma::cube& path,
                       const arma::mat& grid,
                       const arma::imat& control,
                       Rcpp::Function Reward_,
                       const arma::cube& cont,
                       const std::size_t& n_dec,
                       const std::size_t& n_path,
                       const std::size_t& n_dim,
                       const std::size_t& n_pos,
                       const std::size_t& n_action) {
  arma::cube reward(n_path, n_action, n_pos);
  arma::mat compare(n_path, n_action);
  std::size_t nn;
  arma::mat fitted(n_path, n_dim);
  // Find the nearest neighbours for the path nodes
  arma::uvec path_nn(n_path * (n_dec - 1));
  {
    arma::mat nodes(n_path * (n_dec - 1), n_dim);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      nodes.rows(n_path * tt, n_path * (tt + 1) - 1) = path.slice(tt);
    }
    path_nn = arma::conv_to<arma::uvec>::from(rflann::FastKDNeighbour(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(nodes.cols(1, n_dim - 1))),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid.cols(1, n_dim - 1))),
        1)) - 1;
  }
  // Assign decision rules
  arma::uvec host(n_path);
  arma::uvec pos_index = arma::linspace<arma::uvec>(0, n_dim - 1, n_dim);
  for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
    host = path_nn.subvec(n_path * tt, n_path * (tt + 1) - 1);
    reward = Rcpp::as<arma::cube>(Reward_(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t aa = 0; aa < n_action; aa++) {
        nn = control(pp, aa) - 1;
        fitted = cont.slice(tt).submat(host, n_dim * nn + pos_index);
        compare.col(aa) =
            reward.slice(pp).col(aa) + arma::sum(fitted % path.slice(tt), 1);
      }
      policy.slice(tt).col(pp) = arma::index_max(compare, 1);
    }
  }
}

// Partial control and nearest neighbour
void FastOptimalPolicy2(arma::ucube& policy,
                        const arma::cube& path,
                        const arma::mat& grid,
                        const arma::cube& control2,
                        Rcpp::Function Reward_,
                        const arma::cube& cont,
                        const std::size_t& n_dec,
                        const std::size_t& n_path,
                        const std::size_t& n_dim,
                        const std::size_t& n_pos,
                        const std::size_t& n_action) {
  arma::cube reward(n_path, n_action, n_pos);
  arma::mat compare(n_path, n_action);
  arma::mat fitted(n_path, n_dim);
  // Find the nearest neighbours for the path nodes
  arma::uvec path_nn(n_path * (n_dec - 1));
  {
    arma::mat nodes(n_path * (n_dec - 1), n_dim);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      nodes.rows(n_path * tt, n_path * (tt + 1) - 1) = path.slice(tt);
    }
    path_nn = arma::conv_to<arma::uvec>::from(rflann::FastKDNeighbour(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(nodes.cols(1, n_dim - 1))),
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(grid.cols(1, n_dim - 1))),
        1)) - 1;
  }
  // Assign decision rules
  arma::uvec host(n_path);
  arma::uvec pos_index = arma::linspace<arma::uvec>(0, n_dim - 1, n_dim);
  for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
    host = path_nn.subvec(n_path * tt, n_path * (tt + 1) - 1);
    reward = Rcpp::as<arma::cube>(Reward_(
        Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      for (std::size_t aa = 0; aa < n_action; aa++) {
        compare.col(aa) = reward.slice(pp).col(aa);
        for (std::size_t pos = 0; pos < n_pos; pos++) {
          fitted = cont.slice(tt).submat(host, n_dim * pos + pos_index);
          compare.col(aa) +=
              control2(pp, aa, pos) * arma::sum(fitted % path.slice(tt), 1);
        }
      }
      policy.slice(tt).col(pp) = arma::index_max(compare, 1);
    }
  }
}
