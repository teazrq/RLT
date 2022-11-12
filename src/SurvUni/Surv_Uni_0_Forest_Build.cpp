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
                          mat& OOBPrediction,
                          vec& VarImp,
                          mat& AllImp,
                          vec& cindex_tree)
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
  
  // Calculate predictions
  uvec oob_count;
  
  if (do_prediction)
  {
    Prediction.zeros(N, NFail+1);
    OOBPrediction.zeros(N, NFail+1);
    oob_count.zeros(N);
  }
  
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
                                 SURV_FOREST.NodeHazList(nt));
      
      OneTree.initiate(50 + 4*size/nmin);
      
      // build the tree
      if (reinforcement)
      {
        uvec var_protect;
        RLTcout <<"Reinforced survival trees not yet implemented"<<std::endl;
        RLTcout <<"Ignoring command and using standard trees."<<std::endl;
      }

      // const clock_t time_point = clock(); 

      Surv_Uni_Split_A_Node(0, OneTree, SURV_DATA, 
                             Param, inbag_id, var_id, rngl);
      
      // RLTcout << "Core " << omp_get_thread_num() << " Tree " << nt << "; Time Cost: P1 " << 
      //      float(clock() - time_point)/CLOCKS_PER_SEC << std::endl;
      
      // const clock_t begin_time = clock();
      
      // trim tree 
      size_t TreeLength = OneTree.get_tree_length();
      OneTree.trim(TreeLength);

      // RLTcout << "Tree size reduce from " << 50 + 4*size/nmin << " to " << TreeLength << 
      //  " cost time " << 1000 * float( clock() - begin_time ) / CLOCKS_PER_SEC << std::endl;
      
      // inbag and oobag predictions for all subjects
      if (do_prediction)
      {
        uvec proxy_id = linspace<uvec>(0, N-1, N);
        uvec TermNode(N, fill::zeros);
      
        Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, obs_id, TermNode);
      
        for (size_t i = 0; i < N; i++)
          Prediction.row(i) += OneTree.NodeHaz(TermNode(i)).t();
      
        if (oobagObs.n_elem > 0)
        {
          for (size_t i = 0; i < N; i++)
          {
            if (ObsTrack(i, nt) == 0)
            {
              OOBPrediction.row(i) += OneTree.NodeHaz(TermNode(i)).t();
              oob_count(i) += 1;
            }
          }
          
          size_t NTest = oobagObs.n_elem;
          
          uvec oobY = SURV_DATA.Y(oobagObs);
          uvec oobCensor = SURV_DATA.Censor(oobagObs);
          vec oobpred(NTest);
          
          uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
          uvec TermNode(NTest, fill::zeros);
          
          Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, oobagObs, TermNode);
          
          for (size_t i = 0; i < NTest; i++)
            oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) ); // sum of cumulative hazard as prediction
          
          cindex_tree(nt) = 1-cindex_i( oobY, oobCensor, oobpred );
        }
      }
      
      // calculate importance 
      
      if (importance and oobagObs.n_elem > 1)
      {
        uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitVar >= 0 ) ) ));
        
        size_t NTest = oobagObs.n_elem;
        
        uvec oobY = SURV_DATA.Y(oobagObs);
        uvec oobCensor = SURV_DATA.Censor(oobagObs);
        vec oobpred(NTest);
        
        uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
        uvec TermNode(NTest, fill::zeros);
        
        Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, oobagObs, TermNode);
        
        for (size_t i = 0; i < NTest; i++)
          oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) ); // sum of cumulative hazard as prediction
        
        double baseImp = 1-cindex_i( oobY, oobCensor, oobpred );  
        cindex_tree(nt) = 1-cindex_i( oobY, oobCensor, oobpred );
        
        for (size_t j = 0; j < P; j++)
        {
          size_t suffle_var_j = var_id(j);
          
          if (!any(AllVar == suffle_var_j))
            continue;
          
          uvec proxy_id = linspace<uvec>(0, NTest-1, NTest);
          uvec TermNode(NTest, fill::zeros);

          uvec oob_ind = rngl.shuffle(oobagObs);
          vec tildex = SURV_DATA.X.col(suffle_var_j);
          tildex = tildex.elem( oob_ind );  
          
          Find_Terminal_Node_ShuffleJ(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat, proxy_id, oobagObs, TermNode, tildex, suffle_var_j);
          
          // get prediction
          for (size_t i = 0; i < NTest; i++)
            oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) ); // sum of cumulative hazard as prediction
          
          // record
          AllImp(nt, j) =  1-cindex_i( oobY, oobCensor, oobpred) - baseImp;
        }
      }
    }
  }  
  
  if (do_prediction)
  {
    Prediction.shed_col(0);
    Prediction /= ntrees;
    OOBPrediction.shed_col(0);
    for(size_t i = 0; i < N; i++){
      OOBPrediction.row(i)/=oob_count(i);
    }
  }
  
  if (importance){
    VarImp = mean(AllImp, 0).t();
  }
  

}