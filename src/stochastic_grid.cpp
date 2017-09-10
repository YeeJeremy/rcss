// Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Generates a stochastic grid from supplied using k-means clustering
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>

//[[Rcpp::export]]
arma::mat StochasticGrid(const arma::cube& path,
                         const std::size_t& n_grid,
                         const std::size_t& max_iter,
                         const bool& warning) {
  // Prepare the paths
  const std::size_t n_dim = path.n_cols;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dec = path.n_slices;
  arma::mat path_nodes(n_dim, n_path * n_dec);  // Model learn transpose
  for (std::size_t tt = 0; tt < n_dec; tt++) {
    path_nodes.cols(n_path * tt, n_path * (tt + 1) - 1) =
        arma::trans(path.slice(tt));
  }
  // Perform the k-means algorithm
  arma::gmm_diag model;
  model.learn(path_nodes.rows(1, n_dim - 1), n_grid, arma::eucl_dist,
              arma::random_subset, max_iter, 0, 1e-10, warning);
  arma::mat sgrid(n_grid, n_dim);
  sgrid.cols(1, n_dim - 1) = arma::trans(model.means);
  sgrid.col(0).fill(1.0);
  return sgrid;
}
