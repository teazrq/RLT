//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Regression
//  **********************************

// my header file
# include "RLT.h"

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
                               uvec& oob_count,
                               vec& VarImp,
                               vec& VarVI)
{
  // parameters to use
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  size_t P = var_id.n_elem;
  size_t N = obs_id.n_elem;
  size_t size = (size_t) N*Param.resample_prob;
  size_t nmin = Param.nmin;
  size_t importance = Param.importance;
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

  // oob sample too small, turn off importance
  // Only applies to non-replacement sampling (replacement always has ~36.8% OOB)
  if (!replacement && N - size < 2)
    importance = 0;

  // Calculate predictions

  if (importance) do_prediction = true;

  // importance
  mat AllImp;

  if (importance)
    AllImp.zeros(ntrees, P);

  // thread-local storage for OOB accumulation
  std::vector<vec> tl_Prediction(usecores);
  std::vector<uvec> tl_oob_count(usecores);

  // Allocate before entering OpenMP, including embedded forest calls.
  if (do_prediction)
    for (size_t t = 0; t < usecores; t++)
    {
      tl_Prediction[t].zeros(N);
      tl_oob_count[t].zeros(N);
    }
  std::vector<std::exception_ptr> tree_errors(ntrees);

#pragma omp parallel num_threads(usecores)
  {
    int tid = omp_get_thread_num();

    #pragma omp for schedule(dynamic)
    for (size_t nt = 0; nt < ntrees; nt++) // fit all trees
    {
      try
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
        if (Param.useobsweight)
          check_sampled_tree_weight(REG_DATA.obsweight, inbag_id);
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
        uvec var_protect;

        Reg_Uni_Comb_Split_A_Node(0, OneTree, REG_DATA,
                                  Param, inbag_id, var_id, var_protect, rngl);

        // trim tree
        TreeLength = OneTree.get_tree_length();
        OneTree.trim(TreeLength);

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

          // calculate prediction (thread-local)
          tl_Prediction[tid](oobag_index) += OneTree.NodeAve(TermNode);
          tl_oob_count[tid](oobag_index) += 1;
        }

        // calculate permutation importance
        if (importance == 1 and NTest > 1)
        {
          vec oobY = REG_DATA.Y(oobag_id);

          if (TermNode.n_elem == 0){// TermNode not already calculated
            proxy_id = linspace<uvec>(0, NTest-1, NTest);
            TermNode.zeros(NTest);
            Find_Terminal_Node_Comb(0, OneTree, REG_DATA.X, REG_DATA.Ncat,
                                    proxy_id, oobag_id, TermNode);
          }

          // otherwise directly calculate error
          double baseImp = mean(square(oobY - OneTree.NodeAve(TermNode)));

          // what variables are used in this tree
          // Use SplitLoad != 0 to identify actually used variables (consistent with prediction)
          uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitLoad != 0 ) ) ));

          // go through all variables
          for (auto shuffle_var_j : AllVar)
          {
            // reset proxy_id
            proxy_id = linspace<uvec>(0, NTest-1, NTest);

            // create shuffled variable xj
            vec tildex = REG_DATA.X.col(shuffle_var_j);
            tildex = tildex.elem( rngl.shuffle(oobag_id) );

            // find terminal node of shuffled obs
            Find_Terminal_Node_Comb_ShuffleJ(0, OneTree, REG_DATA.X, REG_DATA.Ncat,
                                             proxy_id, oobag_id, TermNode,
                                             tildex, shuffle_var_j);

            // record
            size_t locate_j = find_j(var_id, shuffle_var_j);
            AllImp(nt, locate_j) =  mean(square(oobY - OneTree.NodeAve(TermNode))) - baseImp;
          }
        }


        // probability variable importance
        if (importance == 2 and NTest > 1)
        {
          vec oobY = REG_DATA.Y(oobag_id);

          if (TermNode.n_elem == 0){// TermNode not already calculated
            proxy_id = linspace<uvec>(0, NTest-1, NTest);
            TermNode.zeros(NTest);
            Find_Terminal_Node_Comb(0, OneTree, REG_DATA.X, REG_DATA.Ncat,
                                    proxy_id, oobag_id, TermNode);
          }

          // otherwise directly calculate error
          double baseImp = mean(square(oobY - OneTree.NodeAve(TermNode)));

          // what variables are used in this tree
          // Use SplitLoad != 0 to identify actually used variables (consistent with prediction)
          uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitLoad != 0 ) ) ));
          vec allerror(NTest, fill::zeros);
          vec Prob(TreeLength, fill::zeros);

          // go through all variables
          for (auto randj : AllVar)
          {
            for (size_t i = 0; i < NTest; i ++)
            {
              size_t id = oobag_id(i);
              Prob.zeros();

              Assign_Terminal_Node_Prob_Comb_RandomJ(0,
                                                     OneTree,
                                                     REG_DATA.X,
                                                     REG_DATA.Ncat,
                                                     id,
                                                     1.0,
                                                     Prob,
                                                     randj);

              uvec nonzeronodes = find(Prob > 0);

              // Edge case: no OOB samples for this variable
              if (nonzeronodes.n_elem == 0)
              {
                size_t locate_j = find_j(var_id, randj);
                AllImp(nt, locate_j) = 0;
                continue;
              }

              allerror(i) = accu( Prob(nonzeronodes) % square(OneTree.NodeAve(nonzeronodes) - oobY(i)) );
            }

            size_t locate_j = find_j(var_id, randj);

            // Edge case: all errors are empty (no OOB samples)
            if (allerror.n_elem > 0)
              AllImp(nt, locate_j) = mean(allerror) - baseImp;
            else
              AllImp(nt, locate_j) = 0;
          }
        }
      }
      catch (...)
      {
        tree_errors[nt] = std::current_exception();
      }
    }
  }


  rethrow_tree_build_errors(tree_errors);

  if (do_prediction)
  {
    // merge thread-local results
    Prediction.zeros(N);
    oob_count.zeros(N);
    for (size_t t = 0; t < usecores; t++)
    {
      Prediction += tl_Prediction[t];
      oob_count += tl_oob_count[t];
    }

    // normalize only where oob_count > 0
    uvec valid = find(oob_count > 0);
    Prediction(valid) = Prediction(valid) / oob_count(valid);
  }

  if (importance)
  {
    VarImp = mean(AllImp, 0).t();

    // VI variance estimation (requires var.mode = "matched")
    if (Param.var_mode == 1)
    {
      rowvec Vs = var(AllImp, 1, 0);

      size_t B = ntrees / 2;
      mat Diff = AllImp.rows(0, B-1) - AllImp.rows(B, 2*B-1);
      rowvec Vh = mean(square(Diff), 0) / 2;

      VarVI = (Vh - Vs).t();
    }
  }
}
