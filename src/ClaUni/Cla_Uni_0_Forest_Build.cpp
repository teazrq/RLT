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
  
  if (importance) do_prediction = true;
  
  if (do_prediction)
  {
    Prediction.zeros(N, nclass);
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
        uvec var_protect;
        
        Cla_Uni_Split_A_Node_Embed(0, OneTree, CLA_DATA, 
                                   Param, inbag_id, var_id, var_protect, rngl);
        
      }else{
        Cla_Uni_Split_A_Node(0, OneTree, CLA_DATA, 
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
        proxy_id = linspace<uvec>(0, NTest-1, NTest);
        TermNode.zeros(NTest);
        
        // find terminal codes
        Find_Terminal_Node(0, OneTree, CLA_DATA.X, CLA_DATA.Ncat, 
                           proxy_id, oobag_id, TermNode);
        
        // record terminal node prediction
        mat AllPred(N, nclass);
        for (size_t i = 0; i < NTest; i++)
          AllPred.row(oobag_index(i)) = OneTree.NodeProb.row(TermNode(i));
        
        Prediction += AllPred;
        oob_count(oobag_index) += 1;
      }

      // calculate importance 
      if (importance and NTest > 1)
      {
        uvec oobY = CLA_DATA.Y(oobag_id);
        
        if (TermNode.n_elem == 0){// TermNode not already calculated
          proxy_id = linspace<uvec>(0, NTest-1, NTest);
          TermNode.zeros(NTest);
          Find_Terminal_Node(0, OneTree, CLA_DATA.X, CLA_DATA.Ncat, 
                             proxy_id, oobag_id, TermNode);
        }
        
        // oob prediction error for this tree
        uvec oobpred(NTest);
        
        for (size_t i = 0; i < NTest; i++)
          oobpred(i) = index_max(OneTree.NodeProb.row(TermNode(i)));
        
        double baseImp = (double) sum(oobY != oobpred) / NTest;        
        
        // what variables are used in this tree
        uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitVar >= 0 ) ) ));

        for (auto shuffle_var_j : AllVar)
        {
          uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
          uvec TermNode(NTest, fill::zeros);
          
          // create shuffled variable xj
          vec tildex = CLA_DATA.X.col(shuffle_var_j);
          tildex = tildex.elem( rngl.shuffle(oobag_id) ); 

          // find the terminal of the shuffled data
          Find_Terminal_Node_ShuffleJ(0, OneTree, CLA_DATA.X, CLA_DATA.Ncat, 
                                      proxy_id, oobag_id, TermNode, tildex, shuffle_var_j);
          
          // get prediction error
          for (size_t i = 0; i < NTest; i++)
            oobpred(i) = index_max(OneTree.NodeProb.row(TermNode(i)));
          
          // record
          size_t locate_j = find_j(var_id, shuffle_var_j);
          AllImp(nt, locate_j) = (double) sum(oobY != oobpred) / NTest - baseImp;
        }
      }

      // probability variable importance
      if (importance == 2)
      {
        uvec oobY = CLA_DATA.Y(oobag_id);
        
        if (TermNode.n_elem == 0){// TermNode not already calculated
          proxy_id = linspace<uvec>(0, NTest-1, NTest);
          TermNode.zeros(NTest);
          Find_Terminal_Node(0, OneTree, CLA_DATA.X, CLA_DATA.Ncat, 
                             proxy_id, oobag_id, TermNode);
        }
        
        // oob prediction error for this tree
        uvec oobpred(NTest);
        
        for (size_t i = 0; i < NTest; i++)
          oobpred(i) = index_max(OneTree.NodeProb.row(TermNode(i)));
        
        double baseImp = (double) sum(oobY != oobpred) / NTest; 
        
        // what variables are used in this tree
        uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitVar >= 0 ) ) ));
        vec allerror( NTest, fill::zeros);
        vec Prob(TreeLength, fill::zeros);
        
        // go through all variables
        for (auto randj : AllVar)
        {
          for (size_t i = 0; i < NTest; i ++)
          {
            size_t id = oobag_id(i);            
            Prob.zeros();
            
            Assign_Terminal_Node_Prob_RandomJ(0,
                                              OneTree,
                                              CLA_DATA.X, 
                                              CLA_DATA.Ncat,
                                              id,
                                              1.0,
                                              Prob,
                                              randj);
            
            uvec nonzeronodes = find(Prob > 0);
            mat allprob = OneTree.NodeProb.rows(nonzeronodes);
            uvec label(nonzeronodes.n_elem, fill::zeros);
            
            for (size_t k = 0; k < nonzeronodes.n_elem; k ++)
              label(k) = allprob.row(k).index_max();
            
            allerror(i) = accu( Prob(nonzeronodes) % ( label != oobY(i) ) );
          }
          
          size_t locate_j = find_j(var_id, randj);
          AllImp(nt, locate_j) = mean(allerror) - baseImp;
        }
      }
      

    }
  }
  
  if (do_prediction)
  {
    for (size_t i = 0; i < nclass; i++)
      Prediction.col(i) = Prediction.col(i) / oob_count;
  }
  
  if (importance)
    VarImp = mean(AllImp, 0).t();

}