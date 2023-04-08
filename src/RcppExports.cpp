// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// ClaUniForestFit
List ClaUniForestFit(arma::mat& X, arma::uvec& Y, arma::uvec& Ncat, size_t nclass, arma::vec& obsweight, arma::vec& varweight, arma::imat& ObsTrack, List& param_r);
RcppExport SEXP _RLT_ClaUniForestFit(SEXP XSEXP, SEXP YSEXP, SEXP NcatSEXP, SEXP nclassSEXP, SEXP obsweightSEXP, SEXP varweightSEXP, SEXP ObsTrackSEXP, SEXP param_rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< size_t >::type nclass(nclassSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type obsweight(obsweightSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type varweight(varweightSEXP);
    Rcpp::traits::input_parameter< arma::imat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< List& >::type param_r(param_rSEXP);
    rcpp_result_gen = Rcpp::wrap(ClaUniForestFit(X, Y, Ncat, nclass, obsweight, varweight, ObsTrack, param_r));
    return rcpp_result_gen;
END_RCPP
}
// ClaUniForestPred
List ClaUniForestPred(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeWeight, arma::field<arma::mat>& NodeProb, arma::mat& X, arma::uvec& Ncat, bool VarEst, bool keep_all, size_t usecores, size_t verbose);
RcppExport SEXP _RLT_ClaUniForestPred(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeWeightSEXP, SEXP NodeProbSEXP, SEXP XSEXP, SEXP NcatSEXP, SEXP VarEstSEXP, SEXP keep_allSEXP, SEXP usecoresSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeWeight(NodeWeightSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat>& >::type NodeProb(NodeProbSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< bool >::type VarEst(VarEstSEXP);
    Rcpp::traits::input_parameter< bool >::type keep_all(keep_allSEXP);
    Rcpp::traits::input_parameter< size_t >::type usecores(usecoresSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(ClaUniForestPred(SplitVar, SplitValue, LeftNode, RightNode, NodeWeight, NodeProb, X, Ncat, VarEst, keep_all, usecores, verbose));
    return rcpp_result_gen;
END_RCPP
}
// QuanUniForestFit
List QuanUniForestFit(arma::mat& X, arma::vec& Y, arma::uvec& Ncat, arma::vec& obsweight, arma::vec& varweight, arma::imat& ObsTrack, List& param_r);
RcppExport SEXP _RLT_QuanUniForestFit(SEXP XSEXP, SEXP YSEXP, SEXP NcatSEXP, SEXP obsweightSEXP, SEXP varweightSEXP, SEXP ObsTrackSEXP, SEXP param_rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type obsweight(obsweightSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type varweight(varweightSEXP);
    Rcpp::traits::input_parameter< arma::imat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< List& >::type param_r(param_rSEXP);
    rcpp_result_gen = Rcpp::wrap(QuanUniForestFit(X, Y, Ncat, obsweight, varweight, ObsTrack, param_r));
    return rcpp_result_gen;
END_RCPP
}
// ARMA_EMPTY_UMAT
arma::umat ARMA_EMPTY_UMAT();
RcppExport SEXP _RLT_ARMA_EMPTY_UMAT() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(ARMA_EMPTY_UMAT());
    return rcpp_result_gen;
END_RCPP
}
// ARMA_EMPTY_VEC
arma::vec ARMA_EMPTY_VEC();
RcppExport SEXP _RLT_ARMA_EMPTY_VEC() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(ARMA_EMPTY_VEC());
    return rcpp_result_gen;
END_RCPP
}
// mysample
arma::uvec mysample(size_t Num, size_t min, size_t max, size_t seed);
RcppExport SEXP _RLT_mysample(SEXP NumSEXP, SEXP minSEXP, SEXP maxSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< size_t >::type Num(NumSEXP);
    Rcpp::traits::input_parameter< size_t >::type min(minSEXP);
    Rcpp::traits::input_parameter< size_t >::type max(maxSEXP);
    Rcpp::traits::input_parameter< size_t >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(mysample(Num, min, max, seed));
    return rcpp_result_gen;
END_RCPP
}
// Kernel_Self
List Kernel_Self(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeWeight, arma::mat& X, arma::uvec& Ncat, size_t verbose);
RcppExport SEXP _RLT_Kernel_Self(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeWeightSEXP, SEXP XSEXP, SEXP NcatSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeWeight(NodeWeightSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(Kernel_Self(SplitVar, SplitValue, LeftNode, RightNode, NodeWeight, X, Ncat, verbose));
    return rcpp_result_gen;
END_RCPP
}
// Kernel_Cross
List Kernel_Cross(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeWeight, arma::mat& X1, arma::mat& X2, arma::uvec& Ncat, size_t verbose);
RcppExport SEXP _RLT_Kernel_Cross(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeWeightSEXP, SEXP X1SEXP, SEXP X2SEXP, SEXP NcatSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeWeight(NodeWeightSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X1(X1SEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X2(X2SEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(Kernel_Cross(SplitVar, SplitValue, LeftNode, RightNode, NodeWeight, X1, X2, Ncat, verbose));
    return rcpp_result_gen;
END_RCPP
}
// Kernel_Train
List Kernel_Train(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeWeight, arma::mat& X1, arma::mat& X2, arma::uvec& Ncat, arma::imat& ObsTrack, size_t verbose);
RcppExport SEXP _RLT_Kernel_Train(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeWeightSEXP, SEXP X1SEXP, SEXP X2SEXP, SEXP NcatSEXP, SEXP ObsTrackSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeWeight(NodeWeightSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X1(X1SEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X2(X2SEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::imat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(Kernel_Train(SplitVar, SplitValue, LeftNode, RightNode, NodeWeight, X1, X2, Ncat, ObsTrack, verbose));
    return rcpp_result_gen;
END_RCPP
}
// RegUniCombForestFit
List RegUniCombForestFit(arma::mat& X, arma::vec& Y, arma::uvec& Ncat, arma::vec& obsweight, arma::vec& varweight, arma::imat& ObsTrack, List& param);
RcppExport SEXP _RLT_RegUniCombForestFit(SEXP XSEXP, SEXP YSEXP, SEXP NcatSEXP, SEXP obsweightSEXP, SEXP varweightSEXP, SEXP ObsTrackSEXP, SEXP paramSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type obsweight(obsweightSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type varweight(varweightSEXP);
    Rcpp::traits::input_parameter< arma::imat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< List& >::type param(paramSEXP);
    rcpp_result_gen = Rcpp::wrap(RegUniCombForestFit(X, Y, Ncat, obsweight, varweight, ObsTrack, param));
    return rcpp_result_gen;
END_RCPP
}
// RegUniForestFit
List RegUniForestFit(arma::mat& X, arma::vec& Y, arma::uvec& Ncat, arma::vec& obsweight, arma::vec& varweight, arma::imat& ObsTrack, List& param_r);
RcppExport SEXP _RLT_RegUniForestFit(SEXP XSEXP, SEXP YSEXP, SEXP NcatSEXP, SEXP obsweightSEXP, SEXP varweightSEXP, SEXP ObsTrackSEXP, SEXP param_rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type obsweight(obsweightSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type varweight(varweightSEXP);
    Rcpp::traits::input_parameter< arma::imat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< List& >::type param_r(param_rSEXP);
    rcpp_result_gen = Rcpp::wrap(RegUniForestFit(X, Y, Ncat, obsweight, varweight, ObsTrack, param_r));
    return rcpp_result_gen;
END_RCPP
}
// RegUniForestPred
List RegUniForestPred(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeWeight, arma::field<arma::vec>& NodeAve, arma::mat& X, arma::uvec& Ncat, bool VarEst, bool keep_all, size_t usecores, size_t verbose);
RcppExport SEXP _RLT_RegUniForestPred(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeWeightSEXP, SEXP NodeAveSEXP, SEXP XSEXP, SEXP NcatSEXP, SEXP VarEstSEXP, SEXP keep_allSEXP, SEXP usecoresSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeWeight(NodeWeightSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeAve(NodeAveSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< bool >::type VarEst(VarEstSEXP);
    Rcpp::traits::input_parameter< bool >::type keep_all(keep_allSEXP);
    Rcpp::traits::input_parameter< size_t >::type usecores(usecoresSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(RegUniForestPred(SplitVar, SplitValue, LeftNode, RightNode, NodeWeight, NodeAve, X, Ncat, VarEst, keep_all, usecores, verbose));
    return rcpp_result_gen;
END_RCPP
}
// SurvUniForestFit
List SurvUniForestFit(arma::mat& X, arma::uvec& Y, arma::uvec& Censor, arma::uvec& Ncat, arma::vec& obsweight, arma::vec& varweight, arma::imat& ObsTrack, List& param_r);
RcppExport SEXP _RLT_SurvUniForestFit(SEXP XSEXP, SEXP YSEXP, SEXP CensorSEXP, SEXP NcatSEXP, SEXP obsweightSEXP, SEXP varweightSEXP, SEXP ObsTrackSEXP, SEXP param_rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Censor(CensorSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type obsweight(obsweightSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type varweight(varweightSEXP);
    Rcpp::traits::input_parameter< arma::imat& >::type ObsTrack(ObsTrackSEXP);
    Rcpp::traits::input_parameter< List& >::type param_r(param_rSEXP);
    rcpp_result_gen = Rcpp::wrap(SurvUniForestFit(X, Y, Censor, Ncat, obsweight, varweight, ObsTrack, param_r));
    return rcpp_result_gen;
END_RCPP
}
// SurvUniForestPred
List SurvUniForestPred(arma::field<arma::ivec>& SplitVar, arma::field<arma::vec>& SplitValue, arma::field<arma::uvec>& LeftNode, arma::field<arma::uvec>& RightNode, arma::field<arma::vec>& NodeWeight, arma::field<arma::field<arma::vec>>& NodeHaz, arma::mat& X, arma::uvec& Ncat, size_t& NFail, bool VarEst, bool keep_all, size_t usecores, size_t verbose);
RcppExport SEXP _RLT_SurvUniForestPred(SEXP SplitVarSEXP, SEXP SplitValueSEXP, SEXP LeftNodeSEXP, SEXP RightNodeSEXP, SEXP NodeWeightSEXP, SEXP NodeHazSEXP, SEXP XSEXP, SEXP NcatSEXP, SEXP NFailSEXP, SEXP VarEstSEXP, SEXP keep_allSEXP, SEXP usecoresSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::field<arma::ivec>& >::type SplitVar(SplitVarSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type SplitValue(SplitValueSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type LeftNode(LeftNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::uvec>& >::type RightNode(RightNodeSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec>& >::type NodeWeight(NodeWeightSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::field<arma::vec>>& >::type NodeHaz(NodeHazSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Ncat(NcatSEXP);
    Rcpp::traits::input_parameter< size_t& >::type NFail(NFailSEXP);
    Rcpp::traits::input_parameter< bool >::type VarEst(VarEstSEXP);
    Rcpp::traits::input_parameter< bool >::type keep_all(keep_allSEXP);
    Rcpp::traits::input_parameter< size_t >::type usecores(usecoresSEXP);
    Rcpp::traits::input_parameter< size_t >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(SurvUniForestPred(SplitVar, SplitValue, LeftNode, RightNode, NodeWeight, NodeHaz, X, Ncat, NFail, VarEst, keep_all, usecores, verbose));
    return rcpp_result_gen;
END_RCPP
}
// mc_band
arma::mat mc_band(const arma::vec& mar_sd, const arma::mat& S, const arma::vec& alpha, size_t N);
RcppExport SEXP _RLT_mc_band(SEXP mar_sdSEXP, SEXP SSEXP, SEXP alphaSEXP, SEXP NSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type mar_sd(mar_sdSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< size_t >::type N(NSEXP);
    rcpp_result_gen = Rcpp::wrap(mc_band(mar_sd, S, alpha, N));
    return rcpp_result_gen;
END_RCPP
}
// cindex_d
double cindex_d(arma::vec& Y, arma::uvec& Censor, arma::vec& pred);
RcppExport SEXP _RLT_cindex_d(SEXP YSEXP, SEXP CensorSEXP, SEXP predSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type Censor(CensorSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type pred(predSEXP);
    rcpp_result_gen = Rcpp::wrap(cindex_d(Y, Censor, pred));
    return rcpp_result_gen;
END_RCPP
}
// testcpp
void testcpp(size_t n);
RcppExport SEXP _RLT_testcpp(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< size_t >::type n(nSEXP);
    testcpp(n);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RLT_ClaUniForestFit", (DL_FUNC) &_RLT_ClaUniForestFit, 8},
    {"_RLT_ClaUniForestPred", (DL_FUNC) &_RLT_ClaUniForestPred, 12},
    {"_RLT_QuanUniForestFit", (DL_FUNC) &_RLT_QuanUniForestFit, 7},
    {"_RLT_ARMA_EMPTY_UMAT", (DL_FUNC) &_RLT_ARMA_EMPTY_UMAT, 0},
    {"_RLT_ARMA_EMPTY_VEC", (DL_FUNC) &_RLT_ARMA_EMPTY_VEC, 0},
    {"_RLT_mysample", (DL_FUNC) &_RLT_mysample, 4},
    {"_RLT_Kernel_Self", (DL_FUNC) &_RLT_Kernel_Self, 8},
    {"_RLT_Kernel_Cross", (DL_FUNC) &_RLT_Kernel_Cross, 9},
    {"_RLT_Kernel_Train", (DL_FUNC) &_RLT_Kernel_Train, 10},
    {"_RLT_RegUniCombForestFit", (DL_FUNC) &_RLT_RegUniCombForestFit, 7},
    {"_RLT_RegUniForestFit", (DL_FUNC) &_RLT_RegUniForestFit, 7},
    {"_RLT_RegUniForestPred", (DL_FUNC) &_RLT_RegUniForestPred, 12},
    {"_RLT_SurvUniForestFit", (DL_FUNC) &_RLT_SurvUniForestFit, 8},
    {"_RLT_SurvUniForestPred", (DL_FUNC) &_RLT_SurvUniForestPred, 13},
    {"_RLT_mc_band", (DL_FUNC) &_RLT_mc_band, 4},
    {"_RLT_cindex_d", (DL_FUNC) &_RLT_cindex_d, 3},
    {"_RLT_testcpp", (DL_FUNC) &_RLT_testcpp, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_RLT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
