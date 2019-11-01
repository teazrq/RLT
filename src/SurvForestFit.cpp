//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "RLT.h"
# include "Utility/Utility.h"
# include "survForest.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export()]]
List SurvForestUniFit(arma::mat& X,
          					 arma::uvec& Y,
          					 arma::uvec& Censor,
          					 arma::uvec& Ncat,
          					 List& param,
          					 List& RLTparam,
          					 arma::vec& obsweight,
          					 arma::vec& varweight,
          					 int usecores,
          					 int verbose)
{

  DEBUG_Rcout << "/// THIS IS A DEBUG MODE OF RLT Survival///" << std::endl;
  
  // check number of cores
  usecores = checkCores(usecores, verbose);

  // readin parameters 
  PARAM_GLOBAL Param(param);
  PARAM_RLT Param_RLT;
  
  // create data objects
  size_t N = X.n_rows;
  size_t P = X.n_cols;
  size_t ntrees = Param.ntrees;
  bool kernel_ready = Param.kernel_ready;
  int seed = Param.seed;
    
  // initiate tree and other objects
  std::vector<Surv_Uni_Tree_Class> Forest(ntrees);

  arma::imat ObsTrack(N, ntrees, fill::zeros);

  arma::field<arma::field<arma::uvec>> NodeRegi(ntrees);
  
  vec VarImp(P, fill::zeros);

  // initiate obs id and var id
  uvec obs_id = linspace<uvec>(0, N-1, N);
  uvec var_id = linspace<uvec>(0, P-1, P);
  
  // prediction matrix

  uvec YFail = unique( Y(find(Censor == 1)) );
  size_t NFail = YFail.n_elem;
  
  cube Pred(NFail + 1, ntrees, N, fill::zeros);
  
  // start to fit the model
  Surv_Uni_Forest_Build((const arma::mat&) X,
            					   (const arma::uvec&) Y,
            					   (const arma::uvec&) Censor,
            					   (const arma::uvec&) Ncat,
            					   (const PARAM_GLOBAL&) Param,
            					   (const PARAM_RLT&) Param_RLT,
            					   obsweight,
            					   obs_id,
            					   varweight,
            					   var_id,
            					   Forest,
            					   ObsTrack,
            					   Pred,
            					   NodeRegi,
            					   VarImp,
            					   seed,
            					   usecores,
            					   verbose);

  DEBUG_Rcout << "  --- Finish fitting trees, start saving objects " << std::endl;
  
  // save tree structure to arma::field
  
  List Forest_R = surv_uni_convert_forest_to_r(Forest);
  
  // return subjects to R
  
  List ReturnList;

  ReturnList["FittedForest"] = Forest_R;
  ReturnList["ObsTrack"] = ObsTrack;
  
  if (kernel_ready)
    ReturnList["NodeRegi"] = NodeRegi;
  else
    ReturnList["NodeRegi"] = R_NilValue;
  
  ReturnList["VarImp"] = VarImp;
  
  mat SurvPred(N, Pred.n_rows);
  mat OobSurvPred(N, Pred.n_rows);
  
  for (size_t i=0; i < N; i++)
  {
    SurvPred.row(i) = mean( Pred.slice(i), 1 ).t();
    
    if ( sum(ObsTrack.row(i) == 0) > 0)
      OobSurvPred.row(i) = mean(Pred.slice(i).cols(find(ObsTrack.row(i) == 0)), 1).t();
    else{
      DEBUG_Rcout << "  subject " << i + 1 << " na " << std::endl;
      OobSurvPred.row(i).fill(datum::nan);
    }
      
  }
  
  SurvPred.shed_col(0);
  OobSurvPred.shed_col(0);
  
  ReturnList["Prediction"] = SurvPred;
  ReturnList["OOBPrediction"] = OobSurvPred;

  // c index for model fitting 
  
  uvec nonNAs = find_finite(OobSurvPred.col(0));
  
  ReturnList["cindex"] = datum::nan;
  
  if (nonNAs.n_elem > 2)
  {
      vec oobpred(N, fill::zeros);
      
      for (auto i : nonNAs)
      {
          oobpred(i) = - sum( cumsum( OobSurvPred.row(i) ) ); // sum of cumulative hazard as prediction
      }
      
      uvec oobY = Y(nonNAs);
      uvec oobC = Censor(nonNAs);
      vec oobP = oobpred(nonNAs);
      
      ReturnList["cindex"] =  1- cindex_i( oobY, oobC, oobP );
  }

  return ReturnList;
}



List surv_uni_convert_forest_to_r(std::vector<Surv_Uni_Tree_Class>& Forest)
{
  size_t ntrees = Forest.size();
  
  arma::field<arma::uvec> NodeType_Field(ntrees);
  arma::field<arma::uvec> SplitVar_Field(ntrees);
  arma::field<arma::vec> SplitValue_Field(ntrees);
  arma::field<arma::uvec> LeftNode_Field(ntrees);
  arma::field<arma::uvec> RightNode_Field(ntrees);
  arma::field<arma::field<arma::vec>> NodeHaz_Field(ntrees);
  arma::field<arma::vec> NodeSize_Field(ntrees);
  
  for (size_t nt = 0; nt < ntrees; nt++)
  {
    NodeType_Field[nt] = uvec(Forest[nt].NodeType.begin(), Forest[nt].NodeType.size(), false, true);
    SplitVar_Field[nt] = uvec(Forest[nt].SplitVar.begin(), Forest[nt].SplitVar.size(), false, true);
    SplitValue_Field[nt] = vec(Forest[nt].SplitValue.begin(), Forest[nt].SplitValue.size(), false, true);
    LeftNode_Field[nt] = uvec(Forest[nt].LeftNode.begin(), Forest[nt].LeftNode.size(), false, true);
    RightNode_Field[nt] = uvec(Forest[nt].RightNode.begin(), Forest[nt].RightNode.size(), false, true);
    
    NodeHaz_Field[nt].copy_size(Forest[nt].NodeHaz);
    
    for (size_t j = 0; j < NodeHaz_Field[nt].n_elem; j++)
      NodeHaz_Field[nt][j] = vec(Forest[nt].NodeHaz[j].begin(), Forest[nt].NodeHaz[j].size(), false, true);
    
    NodeSize_Field[nt] = vec(Forest[nt].NodeSize.begin(), Forest[nt].NodeSize.size(), false, true);
  }
  
  return(List::create(Named("NodeType") = NodeType_Field,
                      Named("SplitVar") = SplitVar_Field,
                      Named("SplitValue") = SplitValue_Field,
                      Named("LeftNode") = LeftNode_Field,
                      Named("RightNode") = RightNode_Field,
                      Named("NodeHaz") = NodeHaz_Field,
                      Named("NodeSize") = NodeSize_Field));
}



