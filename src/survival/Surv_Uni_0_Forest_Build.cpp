//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

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
                           uvec& oob_count,
                           vec& VarImp,
                           vec& VarVI)
{
  // parameters to use
  size_t ntrees = Param.ntrees;
  bool replacement = Param.replacement;
  size_t P = var_id.n_elem;
  size_t N = obs_id.n_elem;
  size_t NFail = SURV_DATA.NFail;
  size_t size = (size_t) N*Param.resample_prob;
  size_t nmin = Param.nmin;
  size_t importance = Param.importance;
  size_t usecores = checkCores(Param.ncores, Param.verbose);
  size_t seed = Param.seed;
  bool reinforcement = Param.reinforcement;

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
  // permutation importance needs >= 2 OOB; distributed needs >= 1 OOB
  if (!obs_track_pre && !replacement && importance > 0)
  {
    if (importance == 1 && N - size < 2)
      importance = 0;
    else if (importance == 2 && N - size < 1)
      importance = 0;
  }

  //
  if (importance) do_prediction = true;

  // Requested but unavailable importance is distinct from zero importance.
  if (Param.importance)
  {
    VarImp.set_size(P);
    VarImp.fill(arma::datum::nan);
    if (Param.var_mode == 1)
    {
      VarVI.set_size(P);
      VarVI.fill(arma::datum::nan);
    }
  }

  // importance
  arma::uvec importance_eligible(ntrees, fill::zeros);
  mat AllImp;

  if (importance)
    AllImp.zeros(ntrees, P);

  // thread-local storage for OOB accumulation
  std::vector<mat> tl_Prediction(usecores);
  std::vector<uvec> tl_oob_count(usecores);

  // Allocate before entering OpenMP, including embedded forest calls.
  if (do_prediction)
    for (size_t t = 0; t < usecores; t++)
    {
      tl_Prediction[t].zeros(N, NFail+1);
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
          check_sampled_tree_weight(SURV_DATA.obsweight, inbag_id);
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

          Surv_Uni_Split_A_Node_Embed(0, OneTree, SURV_DATA,
                                      Param, inbag_id, var_id, var_protect, rngl);
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

          // thread-local OOB accumulation
          for (size_t i = 0; i < NTest; i++)
            tl_Prediction[tid].row(oobag_index(i)) += OneTree.NodeHaz(TermNode(i)).t();

          tl_oob_count[tid](oobag_index) += 1;
        }

        // Eligibility is a property of OOB outcomes, including root-only trees.
        bool comparable_oob = false;
        if (importance && NTest > 1)
        {
          uvec oobY = SURV_DATA.Y(oobag_id);
          uvec oobCensor = SURV_DATA.Censor(oobag_id);
          comparable_oob = cindex_has_comparable_pairs(oobY, oobCensor);
        }

        // calculate permutation importance
        if (importance == 1 && comparable_oob)
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
          if (!std::isfinite(baseImp))
            continue;

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
          importance_eligible(nt) = AllImp.row(nt).is_finite();
        }

        // probability variable importance (distribute)
        if (importance == 2 && comparable_oob)
        {
          // oob samples
          uvec oobY = SURV_DATA.Y(oobag_id);
          uvec oobCensor = SURV_DATA.Censor(oobag_id);

          if (TermNode.n_elem == 0){// TermNode not already calculated
            proxy_id = linspace<uvec>(0, NTest-1, NTest);
            TermNode.zeros(NTest);
            Find_Terminal_Node(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat,
                               proxy_id, oobag_id, TermNode);
          }

          // base prediction: sum of cumulative hazard
          vec oobpred(NTest, fill::zeros);
          for (size_t i = 0; i < NTest; i++)
            if (OneTree.NodeHaz(TermNode(i)).n_elem > 0)
              oobpred(i) = accu( cumsum( OneTree.NodeHaz(TermNode(i)) ) );

          // c-index as base error
          double baseImp = cindex_i( oobY, oobCensor, oobpred );
          if (!std::isfinite(baseImp))
            continue;

          // what variables are used in this tree
          uvec AllVar = conv_to<uvec>::from(unique( OneTree.SplitVar( find( OneTree.SplitVar >= 0 ) ) ));
          vec allerror(NTest, fill::zeros);
          vec Prob(TreeLength, fill::zeros);

          // go through all variables
          for (auto randj : AllVar)
          {
            for (size_t i = 0; i < NTest; i++)
            {
              size_t id = oobag_id(i);
              Prob.zeros();

              Assign_Terminal_Node_Prob_RandomJ(0,
                                                OneTree,
                                                SURV_DATA.X,
                                                SURV_DATA.Ncat,
                                                id,
                                                1.0,
                                                Prob,
                                                randj);

              uvec nonzeronodes = find(Prob > 0);

              // weighted CHF: sum over Prob(k) * cumsum(NodeHaz(k))
              double weighted_chf = 0;
              for (auto k : nonzeronodes)
                if (OneTree.NodeHaz(k).n_elem > 0)
                  weighted_chf += Prob(k) * accu( cumsum( OneTree.NodeHaz(k) ) );

              allerror(i) = weighted_chf;
            }

            // c-index for distributed predictions
            size_t locate_j = find_j(var_id, randj);
            AllImp(nt, locate_j) = baseImp - cindex_i( oobY, oobCensor, allerror );
          }
          importance_eligible(nt) = AllImp.row(nt).is_finite();
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
    Prediction.zeros(N, NFail+1);
    oob_count.zeros(N);
    for (size_t t = 0; t < usecores; t++)
    {
      Prediction += tl_Prediction[t];
      oob_count += tl_oob_count[t];
    }

    Prediction.shed_col(0);

    // normalize only where oob_count > 0
    for(size_t i = 0; i < N; i++)
      if (oob_count(i) > 0)
        Prediction.row(i) /= oob_count(i);
  }

  if (importance)
  {
    uvec eligible = find(importance_eligible > 0);
    if (!eligible.is_empty())
      VarImp = mean(AllImp.rows(eligible), 0).t();

    // VI variance estimation (requires var.mode = "matched")
    if (Param.var_mode == 1 && eligible.n_elem == ntrees)
    {
      // V_s: sample variance of per-tree VI (column-wise, N denominator)
      rowvec Vs = var(AllImp, 1, 0);

      // V_h: cross-half difference (first B vs second B trees)
      size_t B = ntrees / 2;
      mat Diff = AllImp.rows(0, B-1) - AllImp.rows(B, 2*B-1);
      rowvec Vh = mean(square(Diff), 0) / 2;

      VarVI = (Vh - Vs).t();
    }
  }
}
