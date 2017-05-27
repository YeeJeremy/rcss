// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// BellmanAccelerated
Rcpp::List BellmanAccelerated(Rcpp::NumericMatrix grid_, Rcpp::NumericVector reward_, Rcpp::NumericVector control_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight, int n_neighbour, Rcpp::Function Neighbour_);
RcppExport SEXP rcss_BellmanAccelerated(SEXP grid_SEXP, SEXP reward_SEXP, SEXP control_SEXP, SEXP disturb_SEXP, SEXP weightSEXP, SEXP n_neighbourSEXP, SEXP Neighbour_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reward_(reward_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< int >::type n_neighbour(n_neighbourSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Neighbour_(Neighbour_SEXP);
    rcpp_result_gen = Rcpp::wrap(BellmanAccelerated(grid_, reward_, control_, disturb_, weight, n_neighbour, Neighbour_));
    return rcpp_result_gen;
END_RCPP
}
// Bellman
Rcpp::List Bellman(Rcpp::NumericMatrix grid_, Rcpp::NumericVector reward_, Rcpp::NumericVector control_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight);
RcppExport SEXP rcss_Bellman(SEXP grid_SEXP, SEXP reward_SEXP, SEXP control_SEXP, SEXP disturb_SEXP, SEXP weightSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reward_(reward_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight(weightSEXP);
    rcpp_result_gen = Rcpp::wrap(Bellman(grid_, reward_, control_, disturb_, weight));
    return rcpp_result_gen;
END_RCPP
}
// Duality
Rcpp::List Duality(Rcpp::NumericVector path_, Rcpp::NumericVector control_, Rcpp::Function Reward_, Rcpp::NumericVector mart_, Rcpp::IntegerVector path_action_);
RcppExport SEXP rcss_Duality(SEXP path_SEXP, SEXP control_SEXP, SEXP Reward_SEXP, SEXP mart_SEXP, SEXP path_action_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type path_(path_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Reward_(Reward_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type mart_(mart_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type path_action_(path_action_SEXP);
    rcpp_result_gen = Rcpp::wrap(Duality(path_, control_, Reward_, mart_, path_action_));
    return rcpp_result_gen;
END_RCPP
}
// ExpectedAccelerated
arma::mat ExpectedAccelerated(Rcpp::NumericMatrix grid_, Rcpp::NumericMatrix value_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight, int n_neighbour, Rcpp::Function Neighbour_);
RcppExport SEXP rcss_ExpectedAccelerated(SEXP grid_SEXP, SEXP value_SEXP, SEXP disturb_SEXP, SEXP weightSEXP, SEXP n_neighbourSEXP, SEXP Neighbour_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type value_(value_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< int >::type n_neighbour(n_neighbourSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Neighbour_(Neighbour_SEXP);
    rcpp_result_gen = Rcpp::wrap(ExpectedAccelerated(grid_, value_, disturb_, weight, n_neighbour, Neighbour_));
    return rcpp_result_gen;
END_RCPP
}
// Expected
arma::mat Expected(Rcpp::NumericMatrix grid_, Rcpp::NumericMatrix value_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight);
RcppExport SEXP rcss_Expected(SEXP grid_SEXP, SEXP value_SEXP, SEXP disturb_SEXP, SEXP weightSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type value_(value_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight(weightSEXP);
    rcpp_result_gen = Rcpp::wrap(Expected(grid_, value_, disturb_, weight));
    return rcpp_result_gen;
END_RCPP
}
// FastBellman
Rcpp::List FastBellman(Rcpp::NumericMatrix grid_, Rcpp::NumericVector reward_, Rcpp::NumericVector control_, Rcpp::IntegerMatrix r_index_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight, Rcpp::Function Neighbour_, int n_smooth, Rcpp::Function SmoothNeighbour_);
RcppExport SEXP rcss_FastBellman(SEXP grid_SEXP, SEXP reward_SEXP, SEXP control_SEXP, SEXP r_index_SEXP, SEXP disturb_SEXP, SEXP weightSEXP, SEXP Neighbour_SEXP, SEXP n_smoothSEXP, SEXP SmoothNeighbour_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type reward_(reward_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type r_index_(r_index_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Neighbour_(Neighbour_SEXP);
    Rcpp::traits::input_parameter< int >::type n_smooth(n_smoothSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type SmoothNeighbour_(SmoothNeighbour_SEXP);
    rcpp_result_gen = Rcpp::wrap(FastBellman(grid_, reward_, control_, r_index_, disturb_, weight, Neighbour_, n_smooth, SmoothNeighbour_));
    return rcpp_result_gen;
END_RCPP
}
// FastExpected
arma::mat FastExpected(Rcpp::NumericMatrix grid_, Rcpp::NumericMatrix value_, Rcpp::IntegerMatrix r_index_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight, Rcpp::Function Neighbour_, int n_smooth, Rcpp::Function SmoothNeighbour_);
RcppExport SEXP rcss_FastExpected(SEXP grid_SEXP, SEXP value_SEXP, SEXP r_index_SEXP, SEXP disturb_SEXP, SEXP weightSEXP, SEXP Neighbour_SEXP, SEXP n_smoothSEXP, SEXP SmoothNeighbour_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type value_(value_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type r_index_(r_index_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Neighbour_(Neighbour_SEXP);
    Rcpp::traits::input_parameter< int >::type n_smooth(n_smoothSEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type SmoothNeighbour_(SmoothNeighbour_SEXP);
    rcpp_result_gen = Rcpp::wrap(FastExpected(grid_, value_, r_index_, disturb_, weight, Neighbour_, n_smooth, SmoothNeighbour_));
    return rcpp_result_gen;
END_RCPP
}
// FastMartingale2
arma::cube FastMartingale2(Rcpp::NumericMatrix grid_, Rcpp::NumericVector value_, Rcpp::NumericVector expected_, Rcpp::NumericVector path_disturb_, Rcpp::IntegerVector path_nn_, Rcpp::Function Neighbour_, Rcpp::NumericVector control_);
RcppExport SEXP rcss_FastMartingale2(SEXP grid_SEXP, SEXP value_SEXP, SEXP expected_SEXP, SEXP path_disturb_SEXP, SEXP path_nn_SEXP, SEXP Neighbour_SEXP, SEXP control_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value_(value_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type expected_(expected_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type path_disturb_(path_disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type path_nn_(path_nn_SEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Neighbour_(Neighbour_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    rcpp_result_gen = Rcpp::wrap(FastMartingale2(grid_, value_, expected_, path_disturb_, path_nn_, Neighbour_, control_));
    return rcpp_result_gen;
END_RCPP
}
// FastMartingale
arma::cube FastMartingale(Rcpp::NumericVector value_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight_, Rcpp::NumericVector path_, Rcpp::IntegerVector path_nn_, Rcpp::Function Neighbour_, Rcpp::NumericMatrix grid_, Rcpp::NumericVector control_);
RcppExport SEXP rcss_FastMartingale(SEXP value_SEXP, SEXP disturb_SEXP, SEXP weight_SEXP, SEXP path_SEXP, SEXP path_nn_SEXP, SEXP Neighbour_SEXP, SEXP grid_SEXP, SEXP control_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value_(value_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight_(weight_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type path_(path_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type path_nn_(path_nn_SEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Neighbour_(Neighbour_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    rcpp_result_gen = Rcpp::wrap(FastMartingale(value_, disturb_, weight_, path_, path_nn_, Neighbour_, grid_, control_));
    return rcpp_result_gen;
END_RCPP
}
// Martingale
arma::cube Martingale(Rcpp::NumericVector value_, Rcpp::NumericVector disturb_, Rcpp::NumericVector weight, Rcpp::NumericVector path_, Rcpp::NumericVector control_);
RcppExport SEXP rcss_Martingale(SEXP value_SEXP, SEXP disturb_SEXP, SEXP weightSEXP, SEXP path_SEXP, SEXP control_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type value_(value_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type path_(path_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    rcpp_result_gen = Rcpp::wrap(Martingale(value_, disturb_, weight, path_, control_));
    return rcpp_result_gen;
END_RCPP
}
// Path
arma::cube Path(Rcpp::NumericVector start_, Rcpp::NumericVector disturb_);
RcppExport SEXP rcss_Path(SEXP start_SEXP, SEXP disturb_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type start_(start_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    rcpp_result_gen = Rcpp::wrap(Path(start_, disturb_));
    return rcpp_result_gen;
END_RCPP
}
// PathPolicy
arma::ucube PathPolicy(Rcpp::NumericVector path_, Rcpp::IntegerVector path_nn_, Rcpp::NumericVector control_, Rcpp::Function Reward_, Rcpp::NumericVector expected_);
RcppExport SEXP rcss_PathPolicy(SEXP path_SEXP, SEXP path_nn_SEXP, SEXP control_SEXP, SEXP Reward_SEXP, SEXP expected_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type path_(path_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type path_nn_(path_nn_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Reward_(Reward_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type expected_(expected_SEXP);
    rcpp_result_gen = Rcpp::wrap(PathPolicy(path_, path_nn_, control_, Reward_, expected_));
    return rcpp_result_gen;
END_RCPP
}
// StochasticGrid
arma::mat StochasticGrid(Rcpp::NumericVector start_, Rcpp::NumericVector disturb_, int n_grid, int max_iter, bool warning);
RcppExport SEXP rcss_StochasticGrid(SEXP start_SEXP, SEXP disturb_SEXP, SEXP n_gridSEXP, SEXP max_iterSEXP, SEXP warningSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type start_(start_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type disturb_(disturb_SEXP);
    Rcpp::traits::input_parameter< int >::type n_grid(n_gridSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type warning(warningSEXP);
    rcpp_result_gen = Rcpp::wrap(StochasticGrid(start_, disturb_, n_grid, max_iter, warning));
    return rcpp_result_gen;
END_RCPP
}
// TestPolicy
arma::vec TestPolicy(int start_position, Rcpp::NumericVector path_, Rcpp::NumericVector control_, Rcpp::Function Reward_, Rcpp::IntegerVector path_action_);
RcppExport SEXP rcss_TestPolicy(SEXP start_positionSEXP, SEXP path_SEXP, SEXP control_SEXP, SEXP Reward_SEXP, SEXP path_action_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type start_position(start_positionSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type path_(path_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Reward_(Reward_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type path_action_(path_action_SEXP);
    rcpp_result_gen = Rcpp::wrap(TestPolicy(start_position, path_, control_, Reward_, path_action_));
    return rcpp_result_gen;
END_RCPP
}
// TestPolicy2
Rcpp::List TestPolicy2(int start_position, Rcpp::NumericVector path_, Rcpp::NumericVector control_, Rcpp::Function Reward_, Rcpp::IntegerVector path_action_);
RcppExport SEXP rcss_TestPolicy2(SEXP start_positionSEXP, SEXP path_SEXP, SEXP control_SEXP, SEXP Reward_SEXP, SEXP path_action_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type start_position(start_positionSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type path_(path_SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type control_(control_SEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type Reward_(Reward_SEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type path_action_(path_action_SEXP);
    rcpp_result_gen = Rcpp::wrap(TestPolicy2(start_position, path_, control_, Reward_, path_action_));
    return rcpp_result_gen;
END_RCPP
}
