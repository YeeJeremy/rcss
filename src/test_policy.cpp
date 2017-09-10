// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Testing the prescibed policy using provided paths
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>

arma::uword NextPosition(const arma::vec &prob_weight) {
  const std::size_t n_pos = prob_weight.n_elem;
  const arma::vec cum_prob = arma::cumsum(prob_weight);
  const double rand_unif = R::runif(0, 1);
  arma::uword next_state = 0;
  for (std::size_t pp = 0; pp < n_pos; pp++) {
    if (rand_unif <= cum_prob(pp)) {
      next_state = pp;
      break;
    }
  }
  return next_state;
}

// Fast testing of policy
//[[Rcpp::export]]
arma::vec TestPolicy(const std::size_t& start_position,
                     const arma::cube& path,
                     Rcpp::NumericVector control_,
                     Rcpp::Function Reward_,
                     Rcpp::Function Scrap_,
                     const arma::ucube& path_action) {
  // R objects into C++
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  const arma::ivec c_dims = control_.attr("dim");
  const std::size_t n_pos = c_dims(0);
  const std::size_t n_action = c_dims(1);
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
  // Testing the policy
  arma::vec value(n_path, arma::fill::zeros);
  arma::uvec pos(n_path);
  pos.fill(start_position - 1);  // Initialise with starting position
  arma::cube reward(n_path, n_action, n_pos);
  arma::uword policy;
  if (full_control) {
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
      for (std::size_t pp = 0; pp < n_path; pp++) {
        policy = path_action(pp, pos(pp), tt) - 1;
        value(pp) += reward(pp, policy, pos(pp));
        pos(pp) = control(pos(pp), policy) - 1;
      }
    }
  } else {
    arma::vec prob_weight(n_pos);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
      for (std::size_t pp = 0; pp < n_path; pp++) {
        policy = path_action(pp, pos(pp), tt) - 1;
        value(pp) += reward(pp, policy, pos(pp));
        prob_weight = control2.tube(pos(pp), policy);
        pos(pp) = NextPosition(prob_weight);
      }
    }
  }
  // Assign scrap reward
  arma::mat scrap(n_path, n_pos);
  scrap = Rcpp::as<arma::mat>(Scrap_(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(n_dec - 1)))));
  for (std::size_t pp = 0; pp < n_path; pp++) {
    value(pp) += scrap(pp, pos(pp));
  }
  return value;
}

// Complete testing of policy
//[[Rcpp::export]]
Rcpp::List FullTestPolicy(const std::size_t& start_position,
                          const arma::cube& path,
                          Rcpp::NumericVector control_,
                          Rcpp::Function Reward_,
                          Rcpp::Function Scrap_,
                          const arma::ucube& path_action) {
  // R objects into C++
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  const arma::ivec c_dims = control_.attr("dim");
  const std::size_t n_pos = c_dims(0);
  const std::size_t n_action = c_dims(1);
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
  // Testing the policy
  arma::mat value(n_path, n_dec);
  arma::umat pos(n_path, n_dec);
  pos.col(0).fill(start_position - 1);  // Initialise with starting position
  arma::cube reward(n_path, n_action, n_pos);
  arma::umat action(n_path, n_dec - 1);
  if (full_control) {
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
      for (std::size_t pp = 0; pp < n_path; pp++) {
        action(pp, tt) = path_action(pp, pos(pp, tt), tt) - 1;
        value(pp, tt) = reward(pp, action(pp, tt), pos(pp, tt));
        pos(pp, tt + 1) = control(pos(pp, tt), action(pp, tt)) - 1;
      }
    }
  } else {
    arma::vec prob_weight(n_pos);
    for (std::size_t tt = 0; tt < (n_dec - 1); tt++) {
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt))), tt + 1));
      for (std::size_t pp = 0; pp < n_path; pp++) {
        action(pp, tt) = path_action(pp, pos(pp, tt), tt) - 1;
        value(pp, tt) = reward(pp, action(pp, tt), pos(pp, tt));
        prob_weight = control2.tube(pos(pp, tt), action(pp, tt));
        pos(pp, tt + 1) = NextPosition(prob_weight);
      }
    }
  }
  // Assign scrap reward
  arma::mat scrap(n_path, n_pos);
  scrap = Rcpp::as<arma::mat>(Scrap_(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(n_dec - 1)))));
  for (std::size_t pp = 0; pp < n_path; pp++) {
    value(pp, n_dec - 1) = scrap(pp, pos(pp, n_dec - 1));
  }
  return Rcpp::List::create(Rcpp::Named("value") = arma::cumsum(value, 1),
                            Rcpp::Named("position") = pos + 1,
                            Rcpp::Named("action") = action + 1);
}
