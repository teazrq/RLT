//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Classification
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Cla_Uni_Forest_Build(const RLT_CLA_DATA& CLA_DATA,
                          Cla_Uni_Forest_Class& CLA_FOREST,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          imat& ObsTrack,
                          bool do_prediction,
                          mat& Prediction,
                          mat& OOBPrediction,
                          vec& VarImp)
{
  // parameters to use
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  size_t P = var_id.n_elem;
  size_t N = obs_id.n_elem;
  size_t size = (size_t) N*Param.resample_prob;
  size_t nmin = Param.nmin;
  bool importance = Param.importance;
  bool reinforcement = Param.reinforcement;
  size_t usecores = checkCores(Param.ncores, Param.verbose);
  size_t seed = Param.seed;
  size_t nclass = CLA_DATA.nclass;

  // set seed
  Rand rng(seed);
  arma::uvec seed_vec = rng.rand_uvec(0, INT_MAX, ntrees);
  
  // track obs matrix
  bool obs_track_pre = false;
  
  if (ObsTrack.n_elem != 0) //if pre-defined
    obs_track_pre = true;
  else
    ObsTrack.zeros(N, ntrees);
  
  // Calculate predictions
  uvec oob_count;
  
  if (do_prediction)
  {
    Prediction.zeros(N, nclass);
    OOBPrediction.zeros(N, nclass);
    oob_count.zeros(N);
  }
  
  // importance
  mat AllImp;
    
  if (importance)
    AllImp.zeros(ntrees, P);
  
  #pragma omp parallel num_threads(usecores)
  {
    #pragma omp for schedule(dynamic)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      // set xoshiro random seed
      Rand rngl(seed_vec(nt));
      
      // get inbag and oobag samples
      uvec inbag_id, oobagObs;

      //If ObsTrack isn't given, set ObsTrack
      if (!obs_track_pre)
        set_obstrack(ObsTrack, nt, size, replacement, rngl);
      
      // Find the samples from pre-defined ObsTrack
      get_samples(inbag_id, oobagObs, obs_id, ObsTrack.unsafe_col(nt));

      // initialize a tree (univariate split)
      Cla_Uni_Tree_Class OneTree(CLA_FOREST.SplitVarList(nt),
                                 CLA_FOREST.SplitValueList(nt),
                                 CLA_FOREST.LeftNodeList(nt),
                                 CLA_FOREST.RightNodeList(nt),
                                 CLA_FOREST.NodeWeightList(nt),
                                 CLA_FOREST.NodeProbList(nt));
      
      size_t TreeLength = 100 + size/nmin*4;
      OneTree.initiate(TreeLength, nclass);

      // build the tree
      if (reinforcement)
      {
        RLTcout << " do reinforcement in classification" << std::endl;
      }else{
        Cla_Uni_Split_A_Node(0, OneTree, CLA_DATA, 
                             Param, inbag_id, var_id, rngl);
      }
      // trim tree 
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      // inbag and oobag predictions for all subjects
      if (do_prediction)
      {
        RLTcout << " do prediction" << std::endl;
      }

      // calculate importance 
      
      if (importance and oobagObs.n_elem > 1)
      {
        RLTcout << " do importance" << std::endl;
      }
    }
  }
  
  if (do_prediction)
  {
    Prediction /= ntrees;
    OOBPrediction = OOBPrediction / oob_count;
  }  
  
  if (importance)
    VarImp = mean(AllImp, 0).t();
}