//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

// Fit function- must be in the main source folder, 
// otherwise Rcpp won't find it

// [[Rcpp::export()]]
List SurvUniForestFit(arma::mat& X,
                      arma::uvec& Y,
                      arma::uvec& Censor,
                      arma::uvec& Ncat,
                      arma::vec& obsweight,
                      arma::vec& varweight,
                      arma::imat& ObsTrack,
                      List& param_r)
{
  
  // reading parameters 
  PARAM_GLOBAL Param;
  Param.PARAM_READ_R(param_r);
  
  if (Param.verbose) Param.print();
  
  size_t NFail = max( Y(find(Censor == 1)) );  
  
  // create data objects  
  RLT_SURV_DATA SURV_DATA(X, Y, Censor, Ncat, NFail, obsweight, varweight);
  
  size_t N = SURV_DATA.X.n_rows;
  size_t P = SURV_DATA.X.n_cols;
  size_t ntrees = Param.ntrees;
  int obs_track = Param.obs_track;
  
  int importance = Param.importance;
  
  // initiate forest argument objects
  arma::field<arma::ivec> SplitVar(ntrees);
  arma::field<arma::vec> SplitValue(ntrees);
  arma::field<arma::uvec> LeftNode(ntrees);
  arma::field<arma::uvec> RightNode(ntrees);
  arma::field<arma::vec> NodeWeight(ntrees);
  arma::field<arma::field<arma::vec>> NodeHaz(ntrees);
  
  //Initiate forest object
  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, 
                                    SplitValue, 
                                    LeftNode, 
                                    RightNode, 
                                    NodeWeight,
                                    NodeHaz);
  
  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  vec cindex_tree;
  cindex_tree.zeros(ntrees);
  
  // Initiate prediction objects
  mat Prediction; // initialization means they will be calculated
  mat OOBPrediction;
  
  // VarImp
  vec VarImp;
  vec VarImpVar;
  if (importance){
    VarImp.zeros(P);
    VarImpVar.zeros(P);
  }
  
  // importance
  mat AllImp;
  
  if (importance)
    AllImp.zeros(ntrees, P);
  
  // Run model fitting
  Surv_Uni_Forest_Build((const RLT_SURV_DATA&) SURV_DATA,
                       SURV_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       (const uvec&) obs_id,
                       (const uvec&) var_id,
                       ObsTrack,
                       true,
                       Prediction,
                       OOBPrediction,
                       VarImp,
                       AllImp,
                       cindex_tree);
  
  //initialize return objects
  List ReturnList;
  ReturnList["NFail"] = NFail;
  
  List Forest_R;
  
  //Save forest objects as part of return list  
  Forest_R["SplitVar"] = SplitVar;
  Forest_R["SplitValue"] = SplitValue;
  Forest_R["LeftNode"] = LeftNode;
  Forest_R["RightNode"] = RightNode;
  Forest_R["NodeWeight"] = NodeWeight;  
  Forest_R["NodeHaz"] = NodeHaz;

  //Add to return list
  ReturnList["FittedForest"] = Forest_R;
  
  if (obs_track) ReturnList["ObsTrack"] = ObsTrack;
  if (importance) ReturnList["VarImp"] = VarImp;
  if (importance) ReturnList["VarImpAll"] = AllImp;
  
  if(any(ObsTrack.col(0)<0)){
    
    size_t B = (size_t) ntrees/2;
    
    uvec firsthalf = linspace<uvec>(0, B-1, B);
    uvec secondhalf = linspace<uvec>(B, 2*B-1, B);
    
    mat AllImpt = AllImp.t();
    
    arma::mat tmp_diff;
    arma::mat Cov_Est(P, P, fill::zeros);

    arma::mat Tree_Cov_Est = Cov_Tree(AllImpt, B);
    
    Cov_Est=cov(AllImpt.t(), 1);

    arma::mat cov = Tree_Cov_Est - Cov_Est;
    ReturnList["VarImpCov"] = cov;
  }
  
  ReturnList["Prediction"] = Prediction;
  ReturnList["OOBPrediction"] = OOBPrediction;
  ReturnList["cindex_tree"] = cindex_tree;
  
  
  // c index for model fitting 
  
  uvec nonNAs = find_finite(OOBPrediction.col(0));
  
  ReturnList["cindex"] = datum::nan;

  if (nonNAs.n_elem > 2)
  {
    vec oobpred(N, fill::zeros);
    
    for (auto i : nonNAs)
    {
      oobpred(i) = sum( cumsum( OOBPrediction.row(i) ) ); // sum of cumulative hazard as prediction
    }
    
    uvec oobY = Y(nonNAs);
    uvec oobC = Censor(nonNAs);
    vec oobP = oobpred(nonNAs);
    
    ReturnList["cindex"] =  cindex_i(oobY, oobC, oobP);
  }
  
  return ReturnList;
}

// [[Rcpp::export()]]
List SurvUniForestPred(arma::field<arma::ivec>& SplitVar,
                       arma::field<arma::vec>& SplitValue,
                       arma::field<arma::uvec>& LeftNode,
                       arma::field<arma::uvec>& RightNode,
                       arma::field<arma::vec>& NodeWeight,
                       arma::field<arma::field<arma::vec>>& NodeHaz,
                       arma::mat& X,
                       arma::uvec& Ncat,
                       size_t& NFail,
                       bool VarEst,
                       bool keep_all,
                       size_t usecores,
                       size_t verbose)
{
  // check number of cores
  usecores = checkCores(usecores, verbose);
  
  // convert R object to forest
  
  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, 
                                    SplitValue, 
                                    LeftNode, 
                                    RightNode, 
                                    NodeWeight,
                                    NodeHaz);
  
  // Initialize prediction objects  
  cube Pred;
  
  // Run prediction
  Surv_Uni_Forest_Pred(Pred,
                      (const Surv_Uni_Forest_Class&) SURV_FOREST,
                      X,
                      Ncat,
                      NFail,
                      usecores,
                      verbose);
  
  // Initialize return list
  List ReturnList;
  
  mat H(Pred.n_slices, Pred.n_rows);
  cube CumPred(size(Pred));
  
#pragma omp parallel num_threads(usecores)
{
  #pragma omp for schedule(static)
  for (size_t i = 0; i < Pred.n_slices; i++)
  {
    H.row(i) = mean(Pred.slice(i), 1).t();
    CumPred.slice(i) = cumsum(Pred.slice(i), 0);
  }
  
}
  
  ReturnList["hazard"] = H; 
  
  mat CumHaz = cumsum(H,1);
  ReturnList["CumHazard"] = CumHaz; 
  
  mat Surv(H);
  vec surv(H.n_rows, fill::ones);
  vec Ones(H.n_rows, fill::ones);
  
  for (size_t j=0; j < Surv.n_cols; j++)
  {
    surv = surv % (Ones - Surv.col(j)); //KM estimator
    Surv.col(j) = surv;
  }
  
  ReturnList["Survival"] = Surv;  
  
  if (keep_all){
    ReturnList["Allhazard"] = Pred;
    ReturnList["AllCHF"] = CumPred;
  }
  
  if (VarEst)
  {

    size_t N = CumPred.n_slices;
    //size_t ntrees = CumPred.n_cols;
    size_t tmpts = CumPred.n_rows;
    size_t B = (size_t) SURV_FOREST.SplitVarList.size()/2;

    arma::mat Tree_Var_Est(N, tmpts, fill::zeros);
    arma::mat tmp_slice;
    arma::mat tmp_diff;
    arma::cube Tree_Cov_Est(tmpts, tmpts, N, fill::zeros);
    arma::cube Cov_Est(tmpts, tmpts, N, fill::zeros);
    arma::mat Var_Est(N, tmpts, fill::zeros);
    
  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(static)
    for(size_t n = 0; n < N; n++){
        Tree_Cov_Est.slice(n)=Cov_Tree(CumPred.slice(n), B);
        Cov_Est.slice(n)=cov(CumPred.slice(n).t(), 1);
      }
  }
    
    arma::cube cov = Tree_Cov_Est - Cov_Est;
    
  #pragma omp parallel num_threads(usecores)
  {
  #pragma omp for schedule(static)
    for(size_t n=0; n<N; n++){
        Var_Est.row(n)=cov.slice(n).diag();
      }
  }

    ReturnList["Cov"] = cov;
    ReturnList["Var"] = Var_Est;
  }
  
  return ReturnList;
  }

arma::mat Cov_Tree(arma::mat& tmp_slice,
                   size_t& B){
  mat tmp_slice2 = tmp_slice.cols(0,B-1)-tmp_slice.cols(B,2*B);
  return tmp_slice2*tmp_slice2.t()/(2*B);
}

// [[Rcpp::export()]]
arma::vec MvnCV(size_t& N, arma::vec& mean_vec, arma::mat& Cov_mat, arma::vec& var_vec){
  arma::mat X = mvnrnd(mean_vec, Cov_mat, N);
  X.clamp(0, arma::datum::inf);
  X.each_col() -= mean_vec;
  X.each_col() /= sqrt(var_vec);
  arma::rowvec cv = max(abs(X),0);
  return(cv.t());
}
