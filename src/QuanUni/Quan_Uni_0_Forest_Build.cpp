//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Quantile
//  **********************************

// my header file
# include "../RLT.h"

using namespace Rcpp;
using namespace arma;

void Quan_Uni_Forest_Build(const RLT_REG_DATA& REG_DATA,
                          Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          const uvec& obs_id,
                          const uvec& var_id,
                          imat& ObsTrack,
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

      // initialize a tree (univariate split)
      Reg_Uni_Tree_Class OneTree(REG_FOREST.SplitVarList(nt),
                                 REG_FOREST.SplitValueList(nt),
                                 REG_FOREST.LeftNodeList(nt),
                                 REG_FOREST.RightNodeList(nt),
                                 REG_FOREST.NodeAveList(nt));
      
      size_t TreeLength = 100 + size/nmin*3;
      OneTree.initiate(TreeLength);

      // build the tree
      if (reinforcement)
      {
        uvec var_protect;
        
        Reg_Uni_Split_A_Node_Embed(0, OneTree, REG_DATA, 
                                   Param, inbag_id, var_id, var_protect, rngl);
      }else{
        Quan_Uni_Split_A_Node(0, OneTree, REG_DATA, 
                             Param, inbag_id, var_id, rngl);
      }
      
      // trim tree 
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      // inbag and oobag predictions for all subjects
      if (do_prediction)
      {
        uvec proxy_id = linspace<uvec>(0, N-1, N);
        uvec TermNode(N, fill::zeros);
      
        Find_Terminal_Node(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, obs_id, TermNode);
      
        vec AllPred = OneTree.NodeAve(TermNode);
      
        Prediction += AllPred;
  
        if (oobagObs.n_elem > 0)
        {
          for (size_t i = 0; i < N; i++)
          {
            if (ObsTrack(i, nt) == 0)
            {
              OOBPrediction(i) += AllPred(i);
              oob_count(i) += 1;
            }
          }
        }
      }

      // calculate importance 
      
      if (importance and oobagObs.n_elem > 1)
      {
        uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitVar >= 0 ) ) ));
        
        size_t NTest = oobagObs.n_elem;
        
        vec oobY = REG_DATA.Y(oobagObs);
        
        uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
        uvec TermNode(NTest, fill::zeros);
        
        Find_Terminal_Node(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, oobagObs, TermNode);
        
        vec oobpred = OneTree.NodeAve(TermNode);
        
        double baseImp = mean(square(oobY - oobpred));
        
        for (size_t j = 0; j < P; j++)
        {
          size_t suffle_var_j = var_id(j);
          
          if (!any(AllVar == suffle_var_j))
            continue;

          uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
          uvec TermNode(NTest, fill::zeros);
          
          uvec oob_ind = rngl.shuffle(oobagObs);
          vec tildex = REG_DATA.X.col(suffle_var_j);
          tildex = tildex.elem( oob_ind );  //shuffle( REG_DATA.X.unsafe_col(j).elem( oobagObs ) );
          
          Find_Terminal_Node_ShuffleJ(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, oobagObs, TermNode, tildex, suffle_var_j);
          
          // get prediction
          vec oobpred = OneTree.NodeAve(TermNode);
          
          // record
          AllImp(nt, j) =  mean(square(oobY - oobpred)) - baseImp;
        }
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