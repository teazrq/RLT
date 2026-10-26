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
                      arma::vec& varprob,
                      arma::imat& ObsTrack,
                      List& param_r)
{
  
  // reading parameters 
  PARAM_GLOBAL Param;
  Param.PARAM_READ_R(param_r);
  
  if (Param.verbose) Param.print();
  
  size_t NFail = max( Y(find(Censor == 1)) );  
  
  // create data objects  
  RLT_SURV_DATA SURV_DATA(X, Y, Censor, Ncat, NFail, obsweight, varprob);
  
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
  uvec oob_count;

  // VarImp
  vec VarImp;
  if (importance)
    VarImp.zeros(P);
  
  // VarVI (VI variance estimation)
  vec VarVI;
  if (importance && Param.var_mode)
    VarVI.zeros(P);
  
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
                       oob_count,
                       VarImp,
                       VarVI);
  
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
  if (importance && Param.var_mode == 1) ReturnList["VarVI"] = VarVI;
  
  if (Prediction.n_elem > 0)
  {
    ReturnList["Prediction"] = Prediction;
  
    // c-index for oob prediction
    // oob sum of cumulative hazard as prediction
    // only compute c-index over observations with oob_count > 0
    uvec valid = find(oob_count > 0);
    vec oobcch(valid.n_elem, fill::zeros);
  
    for (size_t k = 0; k < valid.n_elem; k++)
      oobcch(k) = accu( cumsum( Prediction.row(valid(k)) ) );
    
    uvec subY = Y(valid);
    uvec subCensor = Censor(valid);
    ReturnList["Error"] = 1 - cindex_i(subY, subCensor, oobcch);
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
                       const arma::uvec& mapping_indices,
                       arma::imat& ObsTrack,
                       int var_mode,
                       bool keep_all,
                       size_t usecores,
                       size_t verbose)
{
  usecores = checkCores(usecores, verbose);
  
  Surv_Uni_Forest_Class SURV_FOREST(SplitVar, SplitValue, LeftNode, RightNode, 
                                    NodeWeight, NodeHaz);
  
  size_t N = X.n_rows;
  size_t ntrees = SURV_FOREST.SplitVarList.size();
  bool VarEst = (var_mode > 0);
  
  // New, smaller grid size determined by index vector length
  size_t effective_grid_size = mapping_indices.n_elem;

  umat AllTermNode(N, ntrees, fill::zeros);
  
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

      // Accumulate on the full grid before selecting evaluation times.
      mat pred_i_chf_full = cumsum(pred_i_hazard_full, 1);
      mat pred_i_hazard_reduced = pred_i_hazard_full.cols(mapping_indices);
      mat pred_i_for_var = pred_i_chf_full.cols(mapping_indices);

      Hazard.row(i) = mean(pred_i_hazard_reduced, 0);
      CHazard.row(i) = mean(pred_i_for_var, 0);
      Surv.row(i) = exp(-CHazard.row(i));
      if (keep_all)
        AllHazard.slice(i) = pred_i_hazard_reduced;

      // 5. Variance estimation using standalone functions
      //    Variance is computed on the CHF (cumulative hazard) scale,
      //    not the survival scale. Per Formentini et al. (2022), the
      //    T×T covariance for confidence bands must use CHF values.
      if (VarEst)
      {
        if (var_mode == 1) // matched (debiased matched-sample variance)
        {
          // Split trees into two halves (matched pairs)
          size_t B = ntrees / 2;
          mat Diff = pred_i_for_var.rows(0, B - 1) - pred_i_for_var.rows(B, ntrees - 1);
          mat Vh = Diff.t() * Diff / ntrees;
          mat Vs = cov(pred_i_for_var, 1);
          mat diffmat = Vh - Vs;
          vec eigval;
          mat eigvec;
          eig_sym(eigval, eigvec, diffmat);
          eigval.elem(find(eigval < 1e-6)).fill(1e-6);
          Cov.slice(i) = eigvec * diagmat(eigval) * eigvec.t();
          Cov.slice(i) = (Cov.slice(i) + Cov.slice(i).t()) / 2;
        } else if (var_mode == 2) // IJ
        {
          Cov.slice(i) = compute_surv_ij_variance(pred_i_for_var, ObsTrack);
        } else if (var_mode == 3) // jackknife
        {
          Cov.slice(i) = compute_surv_jack_variance(pred_i_for_var, ObsTrack);
        }
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
  
  return(mar_sd * q.t());
}
