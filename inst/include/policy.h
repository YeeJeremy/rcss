// Copyright <Jeremy Yee> <jeremyyee@outlook.com.au>
// Header file for extracting policies
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_POLICY_H_
#define INST_INCLUDE_POLICY_H_

#include <RcppArmadillo.h>

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
                       const std::size_t& n_action);

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
                        const std::size_t& n_action);

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
                       const std::size_t& n_action);

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
                        const std::size_t& n_action);

#endif  // INST_INCLUDE_POLICY_H_
