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
                       const arma::uvec& mapping_indices, // (New) Index vector from R for filtering
                       bool VarEst,
                       bool keep_all,
                       size_t usecores,
                       size_t verbose)
{
  usecores = checkCores(usecores, verbose);
  
  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, SplitValue, LeftNode, RightNode, 
                                    NodeWeight, NodeHaz);
  
  size_t N = X.n_rows;
  size_t ntrees = SURV_FOREST.SplitVarList.size();
  
  // --- MODIFIED: New, smaller grid size determined by index vector length ---
  size_t effective_grid_size = mapping_indices.n_elem;

  umat AllTermNode(N, ntrees, fill::zeros);
  
  // --- MODIFIED: All output matrices use new, smaller dimensions ---
  mat Hazard(N, effective_grid_size);
  mat CHazard(N, effective_grid_size);
  mat Surv(N, effective_grid_size);
  
  cube Cov;
  if (VarEst)
    Cov.zeros(effective_grid_size, effective_grid_size, N);
  
  cube AllHazard;
  if (keep_all)
    AllHazard.zeros(ntrees, effective_grid_size, N);
  
  
#pragma omp parallel num_threads(usecores)
  {
#pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++)
    {
      uvec proxy_id = linspace<uvec>(0, N - 1, N);
      uvec real_id = linspace<uvec>(0, N - 1, N);
      uvec TermNode(N, fill::zeros);
      Tree_Class OneTree(SURV_FOREST.SplitVarList(nt), SURV_FOREST.SplitValueList(nt),
                         SURV_FOREST.LeftNodeList(nt), SURV_FOREST.RightNodeList(nt),
                         SURV_FOREST.NodeWeightList(nt));
      Find_Terminal_Node(0, OneTree, X, Ncat, proxy_id, real_id, TermNode);
      AllTermNode.col(nt) = TermNode;
    }
    
#pragma omp barrier

    // This part calculates predictions
#pragma omp for schedule(static)
    for (size_t i = 0; i < N; i++)
    {
      // 1. Compute the full hazard matrix for all trees
      mat pred_i_hazard_full(ntrees, NFail);
      for (size_t nt = 0; nt < ntrees; nt++){
        arma::vec H_full = SURV_FOREST.NodeHazList(nt).at(AllTermNode(i, nt));
        pred_i_hazard_full.row(nt) = H_full.subvec(1, NFail).t(); 
      }

      // 2. Compute the full survival matrix for all trees
      mat pred_i_survival_full = cumprod(1 - pred_i_hazard_full, 1); // row-wise cumulative product

      // 3. Subset both matrices using mapping_indices to get reduced matrices
      mat pred_i_hazard_reduced = pred_i_hazard_full.cols(mapping_indices);
      mat pred_i_survival_reduced = pred_i_survival_full.cols(mapping_indices);

      // 4. Assign the mean of the reduced hazard and survival matrices to the output
      Hazard.row(i) = mean(pred_i_hazard_reduced, 0);
      Surv.row(i) = mean(pred_i_survival_reduced, 0);
      CHazard.row(i) = cumsum(Hazard.row(i));

      // 5. For variance estimation, use the reduced survival matrix
      if (VarEst)
      {
        mat pred_i_for_var = pred_i_survival_reduced;
        size_t B = ntrees / 2;
        mat Diff = pred_i_for_var.rows(0, B - 1) - pred_i_for_var.rows(B, ntrees - 1);
        mat Vh = Diff.t() * Diff / ntrees;
        mat Vs = cov(pred_i_for_var, 1);
        vec eigval;
        mat eigvec;
        eig_sym(eigval, eigvec, Vh - Vs);
        eigval.elem(find(eigval < 1e-6)).fill(1e-6);
        Cov.slice(i) = eigvec * diagmat(eigval) * eigvec.t();
        Cov.slice(i) = (Cov.slice(i) + Cov.slice(i).t())/2;
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