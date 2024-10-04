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

  // Initiate prediction objects
  mat Prediction;

  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  
  bool do_prediction = Param.replacement or (Param.resample_prob < 1);

  // Run model fitting
  Surv_Uni_Forest_Build((const RLT_SURV_DATA&) SURV_DATA,
                       SURV_FOREST,
                       (const PARAM_GLOBAL&) Param,
                       (const uvec&) obs_id,
                       (const uvec&) var_id,
                       ObsTrack,
                       do_prediction,
                       Prediction,
                       VarImp);
  
  //initialize return objects
  List ReturnList;
  
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
  
  if (Prediction.n_elem > 0)
  {
    ReturnList["Prediction"] = Prediction;
  
    // c-index for oob prediction
    // uvec nonNAs = find_finite(OOBPrediction.col(0)); if there are nan in prediction
    
    // oob sum of cumulative hazard as prediction
    vec oobcch(N, fill::zeros);
  
    for (size_t i = 0; i < N; i++)
      oobcch(i) = accu( cumsum( Prediction.row(i) ) );
    
    ReturnList["Error"] = 1 - cindex_i(Y, Censor, oobcch);
  }
  
  ReturnList["NFail"] = NFail;
  
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
  
  // parameters
  size_t N = X.n_rows;
  size_t ntrees = SURV_FOREST.SplitVarList.size();
  
  // all terminal nodes
  umat AllTermNode(N, ntrees, fill::zeros);
  
  // predictions
  mat Hazard(N, NFail);
  mat CHazard(N, NFail);
  mat Surv(N, NFail);
  
  cube Cov;

  if (VarEst)
    Cov.zeros(NFail, NFail, N);
  
  cube AllHazard;
  
  if (keep_all)
  {
    AllHazard.zeros(ntrees, NFail, N);
  }
  
  // get terminal node ids for all observations
#pragma omp parallel num_threads(usecores)
  {
#pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      // initiate all observations
      uvec proxy_id = linspace<uvec>(0, N-1, N);
      uvec real_id = linspace<uvec>(0, N-1, N);
      uvec TermNode(N, fill::zeros);
      
      Tree_Class OneTree(SURV_FOREST.SplitVarList(nt),
                         SURV_FOREST.SplitValueList(nt),
                         SURV_FOREST.LeftNodeList(nt),
                         SURV_FOREST.RightNodeList(nt),
                         SURV_FOREST.NodeWeightList(nt));
      
      Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      
      AllTermNode.col(nt) = TermNode;
    }
    
#pragma omp barrier
    
    // calculate prediction for each observations
    
#pragma omp for schedule(static)
    for (size_t i = 0; i < N; i++)
    {
      mat pred_i(ntrees, NFail + 1);

      // get hazard functions of all trees
      for (size_t nt = 0; nt < ntrees; nt++)
        pred_i.row(nt) = SURV_FOREST.NodeHazList(nt).at(AllTermNode(i, nt)).t();
      
      pred_i.shed_col(0);
      
      if (keep_all)
        AllHazard.slice(i) = pred_i;
      
      // get mean hazard, cumulative hazard and survival
      Hazard.row(i) = mean(pred_i, 0);
      CHazard.row(i) = cumsum(Hazard.row(i));
      //Surv.row(i) = exp(- CHazard.row(i));
      Surv.row(i) = cumprod( 1 - Hazard.row(i) );

      // survival of all trees, KM estimate
      pred_i = cumprod(1 - pred_i, 1 ); // change to survival      
      // if we want the variance of CH, then use NA
      // pred_i = cumsum(pred_i, 1);

      // calculate variance
      if (VarEst)
      {
        if (ntrees % 2 == 1){
          RLTcout << "not an even number of trees for variance estimation\n" << std::endl;
        }

        size_t B = ntrees / 2; // ntrees must be an even number

        // tree variance
        mat Diff = pred_i.rows(0, B-1) - pred_i.rows(B, ntrees-1);
        mat Vh = Diff.t() * Diff / ntrees;

        // sample variance
        mat Vs = cov(pred_i, 1); // use 1/N
        
        // correct negative eigen values
        vec eigval;
        mat eigvec;
        eig_sym(eigval, eigvec, Vh - Vs);

        // correct negative eigen values to at least 1e-6
        for (size_t j = 0; j < eigval.n_elem; j++)
          eigval(j) = std::max(eigval(j), 1e-6);

        // reconstruct corrected covariance matrix
        Cov.slice(i) = (eigvec.each_row() % eigval.t()) * eigvec.t();
        Cov.slice(i) = (Cov.slice(i) + Cov.slice(i).t())/2;
        
        // output raw covariance matrix estimation 
        // Cov.slice(i) = Vh - Vs;
        
        // Var.row(i) = diagvec( Cov.slice(i) ).t();
      }
    }
  }
  
  List ReturnList;
  ReturnList["Hazard"] = Hazard;
  ReturnList["CHF"] = CHazard;
  ReturnList["Survival"] = Surv;
  
  if (VarEst)
  {
    ReturnList["Cov"] = Cov;
    // ReturnList["MarVar"] = Var;
  }
  
  if (keep_all)
  {
    ReturnList["AllHazard"] = AllHazard;
  }
  
  return ReturnList;
}

// [[Rcpp::export()]]
arma::mat mc_band(const arma::vec& mar_sd, 
                  const arma::mat& S,
                  const arma::vec& alpha,
                  size_t N)
{
  // generate random samples
  arma::mat X = arma::mvnrnd( zeros( mar_sd.n_elem ), S, N );
  X.each_col() /= mar_sd;
  
  // all cut-offs
  arma::vec cutoffs = max(abs(X),0).t();
  
  arma::vec q = quantile(cutoffs, 1 - alpha);
  
  // RLTcout<< " quantile estimated:" << q << std::endl;
  return(mar_sd * q.t());
}