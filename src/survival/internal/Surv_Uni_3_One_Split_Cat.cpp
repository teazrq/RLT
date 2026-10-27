//  **********************************
//  Reinforcement Learning Trees (RLT)
//  Survival
//  **********************************

// my header file
# include "RLT.h"

using namespace Rcpp;
using namespace arma;
















// find lower or upper bound
void move_cat_index(size_t& lowindex, 
                    size_t& highindex, 
                    std::vector<Surv_Cat_Class>& cat_reduced, 
                    size_t true_cat, 
                    size_t nmin)
{
  // Create a vector of pointers to Cat_Class
  std::vector<Cat_Class*> Categories(true_cat);
  
  for (size_t i = 0; i < true_cat; ++i) {
    Categories[i] = &cat_reduced[i];
  }
  
  // Call the function with the vector of Cat_Class pointers
  move_cat_index(lowindex, highindex, Categories, true_cat, nmin);
}


// record
double record_cat_split(std::vector<Surv_Cat_Class>& cat_reduced,
                        size_t best_cat,
                        size_t true_cat,
                        size_t ncat)
{
  uvec goright(ncat + 1, fill::zeros); // the first element (category) of goright will always be set to 0 --- go left, but this category does not exist.
  
  for (size_t i = 0; i <= best_cat; i++)
    goright[cat_reduced[i].cat] = 0;
  
  for (size_t i = best_cat + 1; i < true_cat; i++)
    goright[cat_reduced[i].cat] = 1;
  
  for (size_t i = true_cat + 1; i < ncat + 1; i++)
    goright[cat_reduced[i].cat] = 0; // for empty category, assign randomly
  
  return pack(ncat + 1, goright);
}


// CoxGrad categorical split
// Uses pseudo-outcome z_eta from Cox gradient, scores partitions with CoxGrad()
void Surv_Uni_CoxGrad_Cat(Split_Class& TempSplit,
                           const uvec& obs_id,
                           const vec& x,
                           const size_t ncat,
                           const uvec& Y, // Y is collapsed
                           const uvec& Censor, // Censor is collapsed
                           const vec& z_eta,
                           const vec& obs_weight,
                           int nsplit,
                           double alpha,
                           bool useobsweight,
                           Rand& rngl)
{
  size_t N = obs_id.n_elem;
  size_t nmin = N * alpha < 1 ? 1 : (size_t)(N * alpha);

  // Build category -> observation index mapping
  std::vector<std::vector<size_t>> cat_obs(ncat + 1);
  std::vector<size_t> cat_count(ncat + 1, 0);

  for (size_t i = 0; i < N; i++)
  {
    size_t temp_cat = (size_t) x(obs_id(i));
    cat_obs[temp_cat].push_back(i);
    cat_count[temp_cat]++;
  }

  // Count non-empty categories
  size_t true_cat = 0;
  for (size_t j = 1; j <= ncat; j++)
    if (cat_count[j] > 0)
      true_cat++;

  if (true_cat <= 1) return;

  // Build compact list of non-empty categories
  uvec cat_list(true_cat);
  std::vector<std::vector<size_t>> cat_list_obs(true_cat);
  size_t idx = 0;
  for (size_t j = 1; j <= ncat; j++)
  {
    if (cat_count[j] > 0)
    {
      cat_list(idx) = j;
      cat_list_obs[idx] = cat_obs[j];
      idx++;
    }
  }

  // Pre-compute sum of (weighted) z_eta per category for scoring
  vec cat_zeta_sum(true_cat, fill::zeros);
  for (size_t k = 0; k < true_cat; k++)
    for (size_t i : cat_list_obs[k])
      cat_zeta_sum(k) += (useobsweight ? obs_weight(i) : 1.0) * z_eta(i);

  // Pre-compute weighted count per category for nmin check
  vec cat_weighted_n(true_cat, fill::zeros);
  for (size_t k = 0; k < true_cat; k++)
    for (size_t i : cat_list_obs[k])
      cat_weighted_n(k) += (useobsweight ? obs_weight(i) : 1.0);

  double best_score = -1;
  double best_cat = 0;
  double temp_score;

  if (nsplit > 0)
  {
    // Random split: random permutation + random cut
    for (int k = 0; k < nsplit; k++)
    {
      vec rand_scores(true_cat);
      for (size_t j = 0; j < true_cat; j++)
        rand_scores(j) = rngl.rand_01();

      uvec perm = stable_sort_index(rand_scores);

      size_t temp_cut = rngl.rand_sizet(0, true_cat - 2);

      // Build Pseudo_X: categories perm(0)..perm(temp_cut) go left
      uvec Pseudo_X(N, fill::zeros);
      double left_w = 0;
      for (size_t j = 0; j <= temp_cut; j++)
      {
        for (size_t i : cat_list_obs[perm(j)])
        {
          Pseudo_X(i) = 1;
          left_w += (useobsweight ? obs_weight(i) : 1.0);
        }
      }

      double total_w = useobsweight ? accu(obs_weight) : (double)N;
      if (left_w < nmin || (total_w - left_w) < nmin) continue;

      if (useobsweight)
        temp_score = CoxGrad_W(Pseudo_X, z_eta, obs_weight);
      else
        temp_score = CoxGrad(Pseudo_X, z_eta);

      if (temp_score > best_score)
      {
        uvec goright(ncat + 1, fill::zeros);
        for (size_t j = temp_cut + 1; j < true_cat; j++)
          goright(cat_list(perm(j))) = 1;
        best_cat = pack(ncat + 1, goright);
        best_score = temp_score;
      }
    }
  }
  else
  {
    // Best split
    if (true_cat > 6)
    {
      // Random 63 binary partitions
      size_t maxrun = 63;
      uvec best_go;

      for (size_t k = 0; k < maxrun; k++)
      {
        uvec temp_go(true_cat, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          temp_go(j) = rngl.rand_01() < 0.5 ? 1 : 0;

        if (temp_go.min() == temp_go.max()) continue;

        double left_w = 0;
        for (size_t j = 0; j < true_cat; j++)
          if (temp_go(j) == 0)
            left_w += cat_weighted_n(j);
        double total_w = accu(cat_weighted_n);
        if (left_w < nmin || (total_w - left_w) < nmin) continue;

        // Score using pre-computed weighted category z_eta sums
        double left_sum = 0;
        for (size_t j = 0; j < true_cat; j++)
          if (temp_go(j) == 0)
            left_sum += cat_zeta_sum(j);

        temp_score = left_sum * left_sum;

        if (temp_score > best_score)
        {
          best_go = temp_go;
          best_score = temp_score;
        }
      }

      if (best_score > 0)
      {
        uvec goright(ncat + 1, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          goright(cat_list(j)) = best_go(j);
        best_cat = pack(ncat + 1, goright);
      }
    }
    else
    {
      // Exhaustive enumeration for K <= 6 via goright_roll_one
      uvec temp_go(true_cat, fill::zeros);
      uvec best_go;

      // start: first category goes right
      temp_go(0) = 1;

      do
      {
        double left_w = 0;
        for (size_t j = 0; j < true_cat; j++)
          if (temp_go(j) == 0)
            left_w += cat_weighted_n(j);
        double total_w = accu(cat_weighted_n);

        if (left_w >= nmin && (total_w - left_w) >= nmin)
        {
          double left_sum = 0;
          for (size_t j = 0; j < true_cat; j++)
            if (temp_go(j) == 0)
              left_sum += cat_zeta_sum(j);

          temp_score = left_sum * left_sum;

          if (temp_score > best_score)
          {
            best_go = temp_go;
            best_score = temp_score;
          }
        }

        goright_roll_one(temp_go);
      } while (temp_go(true_cat - 1) == 0);

      if (best_score > 0)
      {
        uvec goright(ncat + 1, fill::zeros);
        for (size_t j = 0; j < true_cat; j++)
          goright(cat_list(j)) = best_go(j);
        best_cat = pack(ncat + 1, goright);
      }
    }
  }

  if (best_score > TempSplit.score)
  {
    TempSplit.value = best_cat;
    TempSplit.score = best_score;
  }
}









