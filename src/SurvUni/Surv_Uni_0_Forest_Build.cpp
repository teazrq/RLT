//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Forest_Build(const RLT_SURV_DATA& SURV_DATA,
                           Surv_Uni_Forest_Class& SURV_FOREST,
                           const PARAM_GLOBAL& Param,
                           const uvec& obs_id,
                           const uvec& var_id,
                           imat& ObsTrack,
                           bool do_prediction,
                           mat& Prediction,
                           vec& VarImp)
{
  // parameters to use
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  size_t P = var_id.n_elem;
  size_t N = obs_id.n_elem;
  size_t NFail = SURV_DATA.NFail;
  size_t size = (size_t) N*Param.resample_prob;
  size_t nmin = Param.nmin;
  bool importance = Param.importance;
  bool reinforcement = Param.reinforcement;
  size_t usecores = checkCores(Param.ncores, Param.verbose);
  size_t seed = Param.seed;

  // set seed
  Rand rng(seed);
  arma::uvec seed_vec = rng.rand_uvec(0, INT_MAX, ntrees);
  
  // track obs matrix
  bool obs_track_pre = false;
  
  if (ObsTrack.n_elem != 0) //if pre-defined
    obs_track_pre = true;
  else
    ObsTrack.zeros(N, ntrees);
  
  // oob sample too small, turn off importance 
  if (N - size < 2)
    importance = 0;
  
  // 
  if (importance) do_prediction = true;
  
  // Calculate predictions
  uvec oob_count;

  if (do_prediction)
  {
    Prediction.zeros(N, NFail+1);
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
      
      // get inbag and oobag index
      uvec inbag_index, oobag_index;
      
      //If ObsTrack isn't given, set ObsTrack
      if (!obs_track_pre)
        set_obstrack(ObsTrack, nt, size, replacement, rngl);
      
      // Find the samples from pre-defined ObsTrack
      get_index(inbag_index, oobag_index, ObsTrack.unsafe_col(nt));
      uvec inbag_id = obs_id(inbag_index);
      uvec oobag_id = obs_id(oobag_index);

      // sort inbagObs based on Y values
      const uvec& Y = SURV_DATA.Y;
      const uvec& Censor = SURV_DATA.Censor;
      
      std::sort(inbag_id.begin(), inbag_id.end(), [Y, Censor](size_t i, size_t j)
      {
        if (Y(i) == Y(j))
          return(Censor(i) > Censor(j));
        else
          return Y(i) < Y(j);
      });
      
      // initialize a tree (univariate split)
      Surv_Uni_Tree_Class OneTree(SURV_FOREST.SplitVarList(nt),
                                 SURV_FOREST.SplitValueList(nt),
                                 SURV_FOREST.LeftNodeList(nt),
                                 SURV_FOREST.RightNodeList(nt),
                                 SURV_FOREST.NodeWeightList(nt),
                                 SURV_FOREST.NodeHazList(nt));
      
      size_t TreeLength = 100 + size/nmin*6;
      OneTree.initiate(TreeLength);
      
      // build the tree
      if (reinforcement)
      {
        uvec var_protect;
        RLTcout <<"Reinforced survival trees not yet implemented"<<std::endl;
        RLTcout <<"Ignoring command and using standard trees."<<std::endl;
      }else{
        Surv_Uni_Split_A_Node(0, OneTree, SURV_DATA, 
                              Param, inbag_id, var_id, rngl);
      }

      // trim tree 
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      // for predictions
      size_t NTest = oobag_index.n_elem;
      uvec proxy_id;
      uvec TermNode;
      
      // inbag and oobag predictions for all subjects
      if (do_prediction and NTest > 0)
      {
        // objects used for predicting oob samples
        proxy_id = linspace<uvec>(0, NTest-1, NTest);
        TermNode.zeros(NTest);
      
        // find terminal codes
        Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, 
                           proxy_id, oobag_id, TermNode);
      
        for (size_t i = 0; i < NTest; i++)
          Prediction.row(oobag_index(i)) += OneTree.NodeHaz(TermNode(i)).t();
      
        oob_count(oobag_index) += 1;
      }
      
      // calculate importance
      if (importance == 1 and NTest > 1)
      {
        // oob samples
        uvec oobY = SURV_DATA.Y(oobag_id);
        uvec oobCensor = SURV_DATA.Censor(oobag_id);
        vec oobpred(NTest);
        
        if (TermNode.n_elem == 0){// TermNode not already calculated
          proxy_id = linspace<uvec>(0, NTest-1, NTest);
          TermNode.zeros(NTest);
          Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, 
                             proxy_id, oobag_id, TermNode);
        }
        
        // oob sum of cumulative hazard as prediction
        for (size_t i = 0; i < NTest; i++)
          oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) );
        
        // c-index error
        double baseImp = cindex_i( oobY, oobCensor, oobpred );
        
        // what variables are used in this tree
        uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitVar >= 0 ) ) ));
        
        // go through all variables
        for (auto shuffle_var_j : AllVar)
        {
          // reset proxy_id
          proxy_id = linspace<uvec>(0, NTest-1, NTest);
          
          // create shuffled variable xj
          vec tildex = SURV_DATA.X.col(shuffle_var_j);
          tildex = tildex.elem( rngl.shuffle(oobag_id) );
          
          // find terminal node of shuffled obs
          Find_Terminal_Node_ShuffleJ(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, 
                                      proxy_id, oobag_id, TermNode, tildex, shuffle_var_j);
          
          // predicted CCH for permuted data
          for (size_t i = 0; i < NTest; i++)
            oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) ); // sum of cumulative hazard as prediction
          
          // c-index decreasing for permuted data
          size_t locate_j = find_j(var_id, shuffle_var_j);
          AllImp(nt, locate_j) = baseImp - cindex_i( oobY, oobCensor, oobpred );
        }
      }
    }
    
    // #pragma omp barrier
    // calculate things for all observations
    // #pragma omp for schedule(dynamic, 1)
    // for (size_t i = 0; i < N; i++)
    // {    
    //   // save for later
    // }
    
  }
  
  if (do_prediction)
  {
    Prediction.shed_col(0);
    
    for(size_t i = 0; i < N; i++)
      Prediction.row(i)/=oob_count(i);
  }
  
  if (importance)
    VarImp = mean(AllImp, 0).t();

  

}