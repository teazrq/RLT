//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees//Trees.h"
# include "../Utility/Utility.h"
# include "../regForest.h"

#include <xoshiro.h>
#include <dqrng_distribution.h>

using namespace Rcpp;
using namespace arma;

void Reg_Uni_Forest_Build(const RLT_REG_DATA& REG_DATA,
                          Reg_Uni_Forest_Class& REG_FOREST,
                          const PARAM_GLOBAL& Param,
                          const PARAM_RLT& Param_RLT,
                          uvec& obs_id,
                          uvec& var_id,
                          umat& ObsTrack,
                          vec& Prediction,
                          vec& OOBPrediction,
                          vec& VarImp,
                          size_t seed,
                          int usecores,
                          int verbose)
{
  // parameters need to be used
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  double resample_prob = Param.resample_prob;
  size_t P = Param.P;
  size_t N = obs_id.n_elem;
  size_t size = (size_t) obs_id.n_elem*resample_prob;
  size_t nmin = Param.nmin;
  
  bool obs_track_pre = false; 
  
  if (ObsTrack.n_elem != 0)
    obs_track_pre = true; 
  else
    ObsTrack.zeros(N, ntrees);
  
  bool pred = (Prediction.n_elem > 0);       // for Prediction
  bool oob_pred = (OOBPrediction.n_elem > 0);// for OOBPrediction
  int importance = Param.importance;         // for VarImp

  mat Pred;
  
  if (pred or oob_pred)
    Pred.zeros(N, ntrees);
  
  mat AllImp; 
  
  if (importance == 1)
    AllImp = mat(ntrees, P, fill::zeros);
  
  // start parallel trees
    
  // dqrng::xoshiro256plus rng(seed); // properly seeded rng

  #pragma omp parallel num_threads(usecores)
  {
    
    //dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng 
    //lrng.long_jump(omp_get_thread_num() + 1);  // advance rng by 1 ... ncores jumps
    
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      // get inbag and oobag samples

      uvec inbagObs, oobagObs;
      
      if (!obs_track_pre)
        set_obstrack(ObsTrack, nt, size, replacement);
      
      get_samples(inbagObs, oobagObs, obs_id, ObsTrack.unsafe_col(nt));
      
      // initialize a tree (univariate split)

      Reg_Uni_Tree_Class OneTree(REG_FOREST.NodeTypeList(nt), 
                                 REG_FOREST.SplitVarList(nt),
                                 REG_FOREST.SplitValueList(nt),
                                 REG_FOREST.LeftNodeList(nt),
                                 REG_FOREST.RightNodeList(nt),
                                 REG_FOREST.NodeSizeList(nt),
                                 REG_FOREST.NodeAveList(nt));
      
      size_t TreeLength = 1 + size/nmin*3;
      
      OneTree.initiate(TreeLength);

      // start to fit a tree
      OneTree.NodeType(0) = 1; // 0: unused, 1: reserved; 2: internal node; 3: terminal node
      
      Reg_Uni_Split_A_Node(0, OneTree, REG_DATA, 
                           Param, Param_RLT,
                           inbagObs, var_id);
      
      // trim tree 
      TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      // predictions 
      
      if (pred or oob_pred)
      {
        uvec proxy_id = linspace<uvec>(0, N-1, N);
        uvec TermNode(N, fill::zeros);
        
        Uni_Find_Terminal_Node(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, obs_id, TermNode);
        
        Pred.col(nt) = OneTree.NodeAve(TermNode);
      }
      
      if (importance > 0 and oobagObs.n_elem > 1)
      {
        uvec AllVar = unique( OneTree.SplitVar( find( OneTree.NodeType == 2 ) ) );
        
        size_t NTest = oobagObs.n_elem;
        
        vec oobY = REG_DATA.Y(oobagObs);
        
        uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
        uvec TermNode(NTest, fill::zeros);
        
        Uni_Find_Terminal_Node(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, oobagObs, TermNode);
        
        vec oobpred = OneTree.NodeAve(TermNode);
        
        double baseImp = mean(square(oobY - oobpred));
        
        for (auto j : AllVar)
        {
          uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
          uvec TermNode(NTest, fill::zeros);          
          
          vec tildex = shuffle( REG_DATA.X.unsafe_col(j).elem( oobagObs ) );
          
          Uni_Find_Terminal_Node_ShuffleJ(0, OneTree, REG_DATA.X, REG_DATA.Ncat, proxy_id, oobagObs, TermNode, tildex, j);
          
          // get prediction
          vec oobpred = OneTree.NodeAve(TermNode);

          // record
          AllImp(nt, j) =  mean(square(oobY - oobpred)) - baseImp;
        }
      }
    }
    
    #pragma omp barrier
    
    if (importance == 1)
      VarImp = mean(AllImp, 0).t();
    
    if (pred)
      Prediction = mean(Pred, 1);
    
    if (oob_pred)
    {
      #pragma omp for schedule(static)
      for (size_t i = 0; i < N; i++) // fit all trees
      {
        if ( sum(ObsTrack.row(i) == 0) > 0 )
        {
          rowvec OnePred = Pred.row(i);
          OOBPrediction(i) = mean( OnePred( find(ObsTrack.row(i) == 0) ) );
        }else
          OOBPrediction(i) = datum::nan;
      }
    }
  }
}