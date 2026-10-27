//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;

void Surv_Uni_Comb_Forest_Build(const RLT_SURV_DATA& SURV_DATA,
                                Surv_Uni_Comb_Forest_Class& SURV_FOREST,
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
  size_t size = (size_t) N*Param.resample_prob;
  size_t nmin = Param.nmin;
  size_t importance = Param.importance;
  size_t usecores = checkCores(Param.ncores, Param.verbose);
  size_t seed = Param.seed;
  size_t linear_comb = Param.linear_comb;
  size_t NFail = SURV_DATA.NFail;

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
  if (!obs_track_pre && !replacement && N - size < 2)
    importance = 0;

  // Calculate predictions
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
      tl_Prediction[t].zeros(N, NFail + 1);
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

        // sort inbag obs by Y then by censoring (consistent with survival forest)
        uvec sort_idx = stable_sort_index(SURV_DATA.Y(inbag_id));
        inbag_id = inbag_id(sort_idx);

        // sort by censoring within tied Y
        for (size_t i = 0; i < inbag_id.n_elem; )
        {
          size_t j = i;
          while (j < inbag_id.n_elem && SURV_DATA.Y(inbag_id(j)) == SURV_DATA.Y(inbag_id(i)))
            j++;

          if (j - i > 1)
          {
            uvec sub_idx = linspace<uvec>(i, j-1, j-i);
            uvec sub_sort = stable_sort_index(SURV_DATA.Censor(inbag_id(sub_idx)), "descend");
            uvec tmp = inbag_id(sub_idx);
            inbag_id(sub_idx) = tmp(sub_sort);
          }
          i = j;
        }

        // initialize a tree (combination split)
        Surv_Uni_Comb_Tree_Class OneTree(SURV_FOREST.SplitVarList(nt),
                                         SURV_FOREST.SplitLoadList(nt),
                                         SURV_FOREST.SplitValueList(nt),
                                         SURV_FOREST.LeftNodeList(nt),
                                         SURV_FOREST.RightNodeList(nt),
                                         SURV_FOREST.NodeWeightList(nt),
                                         SURV_FOREST.NodeHazList(nt));

        size_t TreeLength = 100 + size/nmin*4;
        OneTree.initiate(TreeLength, linear_comb);

        // build the tree
        uvec var_protect;

        Surv_Uni_Comb_Split_A_Node(0, OneTree, SURV_DATA,
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
          proxy_id = linspace<uvec>(0, NTest-1, NTest);
          TermNode.zeros(NTest);

          Find_Terminal_Node_Comb(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat,
                                  proxy_id, oobag_id, TermNode);

          // accumulate hazard (thread-local)
          for (size_t i = 0; i < NTest; i++)
          {
            vec nodehaz = OneTree.NodeHaz(TermNode(i));

            if (nodehaz.n_elem > 0)
              tl_Prediction[tid].row(oobag_index(i)).cols(0, nodehaz.n_elem-1) += nodehaz.t();
          }

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
          uvec oobY = SURV_DATA.Y(oobag_id);
          uvec oobCensor = SURV_DATA.Censor(oobag_id);

          if (TermNode.n_elem == 0)
          {
            proxy_id = linspace<uvec>(0, NTest-1, NTest);
            TermNode.zeros(NTest);
            Find_Terminal_Node_Comb(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat,
                                    proxy_id, oobag_id, TermNode);
          }

          // base OOB error (1 - c-index)
          vec base_chf(NTest, fill::zeros);
          for (size_t i = 0; i < NTest; i++)
            if (OneTree.NodeHaz(TermNode(i)).n_elem > 0)
              base_chf(i) = accu(cumsum(OneTree.NodeHaz(TermNode(i))));

          double baseImp = 1 - cindex_i(oobY, oobCensor, base_chf);
          if (!std::isfinite(baseImp))
            continue;

          // what variables are used in this tree
          uvec AllVar = conv_to<uvec>::from(unique(OneTree.SplitVar(find(OneTree.SplitLoad != 0))));

          for (auto shuffle_var_j : AllVar)
          {
            proxy_id = linspace<uvec>(0, NTest-1, NTest);
            TermNode.zeros(NTest);

            vec tildex = SURV_DATA.X.col(shuffle_var_j);
            tildex = tildex.elem(rngl.shuffle(oobag_id));

            Find_Terminal_Node_Comb_ShuffleJ(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat,
                                             proxy_id, oobag_id, TermNode,
                                             tildex, shuffle_var_j);

            vec perm_chf(NTest, fill::zeros);
            for (size_t i = 0; i < NTest; i++)
              if (OneTree.NodeHaz(TermNode(i)).n_elem > 0)
                perm_chf(i) = accu(cumsum(OneTree.NodeHaz(TermNode(i))));

            double permImp = 1 - cindex_i(oobY, oobCensor, perm_chf);

            size_t locate_j = find_j(var_id, shuffle_var_j);
            AllImp(nt, locate_j) = permImp - baseImp;
          }
          importance_eligible(nt) = AllImp.row(nt).is_finite();
        }

        // probability variable importance (distribute)
        if (importance == 2 && comparable_oob)
        {
          uvec oobY = SURV_DATA.Y(oobag_id);
          uvec oobCensor = SURV_DATA.Censor(oobag_id);

          if (TermNode.n_elem == 0)
          {
            proxy_id = linspace<uvec>(0, NTest-1, NTest);
            TermNode.zeros(NTest);
            Find_Terminal_Node_Comb(0, OneTree, SURV_DATA.X, SURV_DATA.Ncat,
                                    proxy_id, oobag_id, TermNode);
          }

          // base OOB prediction: sum of cumulative hazard
          vec base_chf(NTest, fill::zeros);
          for (size_t i = 0; i < NTest; i++)
            if (OneTree.NodeHaz(TermNode(i)).n_elem > 0)
              base_chf(i) = accu(cumsum(OneTree.NodeHaz(TermNode(i))));

          double baseImp = cindex_i(oobY, oobCensor, base_chf);
          if (!std::isfinite(baseImp))
            continue;

          // what variables are used in this tree
          uvec AllVar = conv_to<uvec>::from(unique(OneTree.SplitVar(find(OneTree.SplitLoad != 0))));
          vec allerror(NTest, fill::zeros);
          vec Prob(TreeLength, fill::zeros);

          // go through all variables
          for (auto randj : AllVar)
          {
            for (size_t i = 0; i < NTest; i++)
            {
              size_t id = oobag_id(i);
              Prob.zeros();

              Assign_Terminal_Node_Prob_Comb_RandomJ(0,
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
                  weighted_chf += Prob(k) * accu(cumsum(OneTree.NodeHaz(k)));

              allerror(i) = weighted_chf;
            }

            size_t locate_j = find_j(var_id, randj);
            AllImp(nt, locate_j) = baseImp - cindex_i(oobY, oobCensor, allerror);
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
    Prediction.zeros(N, NFail + 1);
    oob_count.zeros(N);
    for (size_t t = 0; t < usecores; t++)
    {
      Prediction += tl_Prediction[t];
      oob_count += tl_oob_count[t];
    }

    Prediction.shed_col(0);

    // normalize only where oob_count > 0
    uvec valid = find(oob_count > 0);
    for (size_t i = 0; i < valid.n_elem; i++)
    {
      Prediction.row(valid(i)) /= oob_count(valid(i));
    }
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
