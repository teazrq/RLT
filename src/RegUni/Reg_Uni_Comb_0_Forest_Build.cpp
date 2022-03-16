//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
  
using namespace Rcpp;
using namespace arma;

void Reg_Uni_Comb_Forest_Build(const RLT_REG_DATA& REG_DATA,
                            Reg_Uni_Comb_Forest_Class& REG_FOREST,
                            const PARAM_GLOBAL& Param,
                            const uvec& obs_id,
                            const uvec& var_id,
                            umat& ObsTrack,
                            bool do_prediction,
                            vec& Prediction,
                            vec& OOBPrediction,
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
  size_t linear_comb = Param.linear_comb;

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
    Prediction.zeros(N);
    OOBPrediction.zeros(N);
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
      
      // initialize a tree (combination split)      
      
      Reg_Uni_Comb_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                      REG_FOREST.SplitLoadList(nt),
                                      REG_FOREST.SplitValueList(nt),
                                      REG_FOREST.LeftNodeList(nt),
                                      REG_FOREST.RightNodeList(nt),
                                      REG_FOREST.NodeAveList(nt));
      
      size_t TreeLength = 3 + size/nmin*3;
      OneTree.initiate(TreeLength, linear_comb);
      
      // build the tree
      if (reinforcement)
      {
        uvec var_protect;
        
        Reg_Uni_Comb_Split_A_Node_Embed(0, OneTree, REG_DATA, 
                                        Param, inbag_id, var_id, var_protect, rngl);
      }else{
        Reg_Uni_Comb_Split_A_Node(0, OneTree, REG_DATA, 
                                  Param, inbag_id, var_id, rngl);
      }
      
      // trim tree
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      Rcout << "print tree ..." << std::endl;
      
      OneTree.print();

    }
  }
}