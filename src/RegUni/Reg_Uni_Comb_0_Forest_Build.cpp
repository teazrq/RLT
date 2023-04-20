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
                               imat& ObsTrack,
                               bool do_prediction,
                               vec& Prediction,
                               vec& VarImp)
{
  // parameters to use
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  size_t P = var_id.n_elem;
  size_t N = obs_id.n_elem;
  size_t size = (size_t) N*Param.resample_prob;
  size_t nmin = Param.nmin;
  size_t importance = Param.importance;
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
  
  if (importance) do_prediction = true;
  
  if (do_prediction)
  {
    Prediction.zeros(N);
    oob_count.zeros(N);
  }
  
  // importance
  mat AllImp;
    
  if (importance)
    AllImp.zeros(ntrees, P);

//  #pragma omp parallel num_threads(usecores)
  {
//    #pragma omp for schedule(dynamic)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      // set xoshiro random seed
      Rand rngl(seed_vec(nt));
      
      // get inbag and oobag index
      uvec inbag_index, oobag_index;
      
      //If ObsTrack isn't given, set ObsTrack
      if (!obs_track_pre)
        set_obstrack(ObsTrack, nt, size, replacement, rngl);
      
      // Find the samples from pre-defined ObsTrack
      get_index(inbag_index, oobag_index, ObsTrack.unsafe_col(nt));
      uvec inbag_id = obs_id(inbag_index);
      uvec oobag_id = obs_id(oobag_index);
      
      // initialize a tree (combination split)      
      
      Reg_Uni_Comb_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                      REG_FOREST.SplitLoadList(nt),
                                      REG_FOREST.SplitValueList(nt),
                                      REG_FOREST.LeftNodeList(nt),
                                      REG_FOREST.RightNodeList(nt),
                                      REG_FOREST.NodeWeightList(nt),
                                      REG_FOREST.NodeAveList(nt));
      
      size_t TreeLength = 100 + size/nmin*4;
      OneTree.initiate(TreeLength, linear_comb);
      
      // build the tree
      uvec var_protect; // need to get this done

      Reg_Uni_Comb_Split_A_Node(0, OneTree, REG_DATA, 
                                Param, inbag_id, var_id, rngl);

      // trim tree
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      //RLTcout << "-------- print tree ----------" << std::endl;
      //OneTree.print();
      //RLTcout << OneTree.SplitVar << "\n";
      //RLTcout << OneTree.SplitLoad << "\n";
      
      // for predictions
      size_t NTest = oobag_index.n_elem;
      uvec proxy_id;
      uvec TermNode;
      
      // oobag prediction
      if (do_prediction and NTest > 0)
      {
        // objects used for predicting oob samples
        proxy_id = linspace<uvec>(0, NTest-1, NTest);
        TermNode.zeros(NTest);
        
        // find terminal codes
        Find_Terminal_Node_Comb(0, OneTree, REG_DATA.X, REG_DATA.Ncat, 
                                proxy_id, oobag_id, TermNode);
        
        // calculate prediction
        Prediction(oobag_index) += OneTree.NodeAve(TermNode);
        oob_count(oobag_index) += 1;
      }


      
    }
  }
  
  
  if (do_prediction)
    Prediction = Prediction / oob_count;
  
  if (importance)
    VarImp = mean(AllImp, 0).t();
}