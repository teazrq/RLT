//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Univariate Survival 
//  **********************************

// my header file
# include "../RLT.h"
# include "../Trees/Trees.h"
# include "../Utility/Utility.h"
# include "../survForest.h"

#include <xoshiro.h>
#include <dqrng_distribution.h>

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Forest_Build(const RLT_SURV_DATA& SURV_DATA,
                           Surv_Uni_Forest_Class& SURV_FOREST,
                           const PARAM_GLOBAL& Param,
                           const PARAM_RLT& Param_RLT,
                           uvec& obs_id,
                           uvec& var_id,
                           umat& ObsTrack,
                           mat& Prediction,
                           mat& OOBPrediction,
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
  size_t NFail = SURV_DATA.NFail;
  size_t size = (size_t) obs_id.n_elem*resample_prob;
  size_t nmin = Param.nmin;
  
  bool obs_track_pre = false; 
  
  if (ObsTrack.n_elem != 0)
    obs_track_pre = true; 
  else
    ObsTrack.zeros(N, ntrees);
  
  // predictions
  bool pred_cal = true; // this could be changed later to an argument
  
  if (pred_cal)
    Prediction.zeros(N, NFail+1);
  
  bool oob_pred_cal = (replacement or resample_prob < 1);
  uvec oob_count;
  
  if (oob_pred_cal)
  {
    OOBPrediction.zeros(N, NFail+1);
    oob_count.zeros(N);
  }
  
  // importance
  
  int importance = Param.importance;
  
  mat AllImp;
  
  if (importance == 1)
    AllImp = mat(ntrees, P, fill::zeros);
  
  // start parallel trees

  // dqrng::xoshiro256plus rng(seed); // properly seeded rng
    
  #pragma omp parallel num_threads(usecores)
  {
    
    // dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng 
    // lrng.long_jump(omp_get_thread_num() + 1);  // advance rng by 1 ... ncores jumps
      
    #pragma omp for schedule(static)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      // get inbag and oobag samples
      
      uvec inbagObs, oobagObs;
      
      if (!obs_track_pre)
        set_obstrack(ObsTrack, nt, size, replacement);
      
      get_samples(inbagObs, oobagObs, obs_id, ObsTrack.unsafe_col(nt));
      
      // sort inbagObs based on Y values
      const uvec& Y = SURV_DATA.Y;
      const uvec& Censor = SURV_DATA.Censor;

      std::sort(inbagObs.begin(), inbagObs.end(), [Y, Censor](size_t i, size_t j)
        {
          if (Y(i) == Y(j))
            return(Censor(i) > Censor(j));
          else
            return Y(i) < Y(j);
        });
      
      // initialize a tree (univariate split)
      
      Surv_Uni_Tree_Class OneTree(SURV_FOREST.NodeTypeList(nt), 
                                  SURV_FOREST.SplitVarList(nt),
                                  SURV_FOREST.SplitValueList(nt),
                                  SURV_FOREST.LeftNodeList(nt),
                                  SURV_FOREST.RightNodeList(nt),
                                  SURV_FOREST.NodeSizeList(nt),
                                  SURV_FOREST.NodeHazList(nt));

      size_t TreeLength = 3 + size/nmin*3;      

      OneTree.initiate(TreeLength);
      
      // start to fit a tree
      OneTree.NodeType(0) = 1; // 0: unused, 1: reserved; 2: internal node; 3: terminal node
      
      Surv_Uni_Split_A_Node(0, OneTree, SURV_DATA,
                            Param, Param_RLT,
                            inbagObs, var_id);
      
      TreeLength = OneTree.get_tree_length();

      OneTree.trim(TreeLength);
      
      // predictions for all subjects
      
      if (pred_cal and oob_pred_cal)
      {
        uvec proxy_id = linspace<uvec>(0, N-1, N);
        uvec TermNode(N, fill::zeros);
        
        Uni_Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, obs_id, TermNode);        
        
        for (size_t i = 0; i < N; i++)
          Prediction.row(i) += OneTree.NodeHaz(TermNode(i)).t();
        
        for (auto i : oobagObs)
          OOBPrediction.row(i) += OneTree.NodeHaz(TermNode(i)).t();

        oob_count(oobagObs) += 1;
        
      }
      
      // predictions for oob data only 
      
      if (!pred_cal and oob_pred_cal)
      {
        size_t NTest = oobagObs.n_elem;
        uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
        uvec TermNode(NTest, fill::zeros);
        Uni_Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, oobagObs, TermNode);

        for (size_t i = 0; i < NTest; i++)
          OOBPrediction.row(oobagObs(i)) += OneTree.NodeHaz(TermNode(i)).t();
        
        oob_count(oobagObs) += 1;
      }
      
      // calculate importance 
      
      if (importance > 0 and oobagObs.n_elem > 1)
      {
        uvec AllVar = unique( OneTree.SplitVar( find( OneTree.NodeType == 2 ) ) );
        
        size_t NTest = oobagObs.n_elem;
        
        DEBUG_Rcout << "-- calculate variable importance on " << AllVar.n_elem << " variables " << std::endl;
        
        uvec oobY = Y(oobagObs);
        uvec oobC = Censor(oobagObs);
        
        uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
        uvec TermNode(NTest, fill::zeros);
        
        Uni_Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, oobagObs, TermNode);
        
        vec oobpred(NTest, fill::zeros);
        
        for (size_t i =0; i < NTest; i++)
        {
          oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) ); // sum of cumulative hazard as prediction
        }
        
        double baseImp = cindex_i( oobY, oobC, oobpred );  
        
        //for (size_t j = 0; j < P; j ++)  
        for (auto j : AllVar)
        {
          uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);

          vec tildex = shuffle( SURV_DATA.X.unsafe_col(j).elem( oobagObs ) );
          
          Uni_Find_Terminal_Node_ShuffleJ(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, oobagObs, TermNode, tildex, j);
          
          // get prediction
          for (size_t i =0; i < NTest; i++)
          {
            oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) ); // sum of cumulative hazard as prediction
          }

          // record 
          
          AllImp(nt, j) =  baseImp - cindex_i( oobY, oobC, oobpred);
        }
      }
    }
  }
    
  if (importance == 1)
  {
    VarImp = mean(AllImp, 0).t();
  }

  if (pred_cal)
  {
    Prediction.shed_col(0);    
    Prediction /= ntrees; 
  }
  
  if (oob_pred_cal)
  {
    OOBPrediction.shed_col(0);    
    
    for (size_t j = 0; j < NFail; j++)
    {
      OOBPrediction.col(j) = OOBPrediction.col(j) / oob_count;
    }
  }
}