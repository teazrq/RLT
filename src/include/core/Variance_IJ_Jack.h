#ifndef VARIANCE_IJ_JACK_H
#define VARIANCE_IJ_JACK_H

#include <RcppArmadillo.h>
#include <cmath>

namespace {

// Compute infinitesimal jackknife variance for random forest predictions
//
// @param PredAll   n_test x B matrix of per-tree predictions
// @param ObsTrack  n_train x B matrix of inbag counts
// @return          n_test vector of variance estimates
inline arma::vec compute_ij_variance(const arma::mat& PredAll, const arma::imat& ObsTrack)
{
    size_t B = PredAll.n_cols;
    size_t n_train = ObsTrack.n_rows;

    // Clean negative values in ObsTrack
    arma::imat ObsTrack_clean = ObsTrack;
    arma::uvec neg_idx = arma::find(ObsTrack_clean < 0);
    if (!neg_idx.is_empty()) {
        ObsTrack_clean.elem(neg_idx).zeros();
    }

    // Center predictions: subtract row means
    arma::mat pred_centered = PredAll - arma::repmat(arma::mean(PredAll, 1), 1, B);

    // Center inbag counts per observation
    arma::mat counts = arma::conv_to<arma::mat>::from(ObsTrack_clean);
    arma::vec row_means_ot = arma::mean(counts, 1);
    arma::mat N_centered = counts - arma::repmat(row_means_ot, 1, B);

    // Compute raw IJ:
    // V_IJ = (1/B^2) * sum_i (sum_b N_c[i,b] * pred_c[j,b])^2
    // Efficient matrix form using NTN = N_c' %*% N_c  (B x B)
    // PNTN = pred_c %*% NTN  (n_test x B)
    // raw_IJ = rowSums(pred_c * PNTN) / B^2  (element-wise multiplication)
    arma::mat NTN = N_centered.t() * N_centered;   // B x B
    arma::mat PNTN = pred_centered * NTN;          // n_test x B
    arma::vec raw_IJ = arma::sum(pred_centered % PNTN, 1) / (B * B);

    // Monte Carlo bias correction
    arma::vec row_means_N = arma::mean(counts, 1);
    arma::vec row_means_sq_N = arma::mean(arma::square(counts), 1);
    double N_var = arma::mean(row_means_sq_N - arma::square(row_means_N));
    arma::vec boot_var = arma::sum(arma::square(pred_centered), 1) / B;
    arma::vec bias_corr = n_train * N_var * boot_var / B;

    // Final variance (floor at 0)
    arma::vec var_hat = raw_IJ - bias_corr;
    arma::uvec neg_var = arma::find(var_hat < 0);
    if (!neg_var.is_empty()) {
        var_hat.elem(neg_var).zeros();
    }

    return var_hat;
}

// Compute jackknife-after-bootstrap variance for random forest predictions
//
// @param PredAll   n_test x B matrix of per-tree predictions
// @param ObsTrack  n_train x B matrix of inbag counts
// @return          n_test vector of variance estimates
inline arma::vec compute_jack_variance(const arma::mat& PredAll, const arma::imat& ObsTrack)
{
    size_t B = PredAll.n_cols;
    size_t n_test = PredAll.n_rows;

    // Clean negative values in ObsTrack
    arma::imat ObsTrack_clean = ObsTrack;
    arma::uvec neg_idx = arma::find(ObsTrack_clean < 0);
    if (!neg_idx.is_empty()) {
        ObsTrack_clean.elem(neg_idx).zeros();
    }

    // OOB indicator matrix
    arma::umat oob = (ObsTrack_clean == 0);

    // Forest prediction (average over trees)
    arma::vec y_hat = arma::mean(PredAll, 1);

    // OOB count per training observation
    arma::uvec oob_count = arma::sum(oob, 1);
    arma::uvec valid = arma::find(oob_count > 0);

    if (valid.is_empty()) {
        Rcpp::warning("No OOB observations found for jackknife variance. Consider increasing ntrees.");
        return arma::vec(n_test, arma::fill::value(arma::datum::nan));
    }

    size_t n_valid = valid.n_elem;

    // Compute jackknife predictions:
    // For each training obs i, average PredAll over trees where i is OOB
    // jack_pred = PredAll %*% t(oob[valid, ])  -> n_test x n_valid
    arma::mat jack_pred = PredAll * oob.rows(valid).t();
    for (size_t j = 0; j < n_valid; j++) {
        jack_pred.col(j) /= oob_count(valid[j]);
    }

    // Jackknife variance: V_J = (n_valid-1)/n_valid * sum_i (T_bar_i - y_hat)^2
    arma::vec V_J(n_test, arma::fill::zeros);
    for (size_t j = 0; j < n_valid; j++) {
        arma::vec diff = jack_pred.col(j) - y_hat;
        V_J += arma::square(diff);
    }
    V_J *= (static_cast<double>(n_valid) - 1.0) / static_cast<double>(n_valid);

    // Bias correction: (e - 1) * n_valid / B^2 * sum_b (PredAll[,b] - y_hat)^2
    arma::vec bias(n_test, arma::fill::zeros);
    for (size_t b = 0; b < B; b++) {
        arma::vec diff = PredAll.col(b) - y_hat;
        bias += arma::square(diff);
    }
    bias *= (std::exp(1.0) - 1.0) * static_cast<double>(n_valid) / (B * B);

    // Final variance (floor at 0)
    arma::vec var_hat = V_J - bias;
    arma::uvec neg_var = arma::find(var_hat < 0);
    if (!neg_var.is_empty()) {
        var_hat.elem(neg_var).zeros();
    }

    return var_hat;
}

// ============================================================
// Regression variance: matched-sample U-statistic
// ============================================================

// Compute matched-sample variance for random forest predictions
//
// @param PredAll   n_test x 2B matrix of per-tree predictions (first B paired with second B)
// @return          n_test vector of variance estimates
inline arma::vec compute_matched_variance(const arma::mat& PredAll)
{
    size_t B = PredAll.n_cols / 2;

    arma::uvec firsthalf = arma::linspace<arma::uvec>(0, B-1, B);
    arma::uvec secondhalf = arma::linspace<arma::uvec>(B, 2*B-1, B);

    // Vs = sample variance of tree predictions (using N as denominator)
    arma::vec Vs = arma::var(PredAll, 1, 1);

    // Vh = half the mean squared paired difference
    arma::mat TreeDiff = PredAll.cols(firsthalf) - PredAll.cols(secondhalf);
    arma::vec Vh = arma::mean(arma::square(TreeDiff), 1) / 2;

    // Variance = Vh - Vs (can be negative → floor at 0)
    arma::vec Var = Vh - Vs;
    arma::uvec neg_var = arma::find(Var < 0);
    if (!neg_var.is_empty())
        Var.elem(neg_var).zeros();

    return Var;
}

// ============================================================
// Classification variance: cube-based (per-class) methods
// ============================================================

// Classification matched-sample variance
//
// @param PredAll   ntrees x nclass x N cube of per-tree class probabilities
// @param N         number of test observations
// @param nclass    number of classes
// @param ntrees    number of trees (must be 2*B, paired)
// @return          N x nclass matrix of variance estimates
inline arma::mat compute_cla_matched_variance(const arma::cube& PredAll,
                                               size_t N,
                                               size_t nclass,
                                               size_t ntrees)
{
    size_t B = ntrees / 2;
    arma::mat Var(N, nclass, arma::fill::zeros);

    for (size_t i = 0; i < N; i++)
    {
        arma::rowvec Vs = arma::var(PredAll.slice(i), 1, 0);

        arma::mat TreeDiff = PredAll.slice(i).rows(0, B-1) - PredAll.slice(i).rows(B, 2*B-1);
        arma::rowvec Vh = arma::mean(arma::square(TreeDiff), 0) / 2;

        arma::rowvec v = Vh - Vs;
        arma::uvec neg = arma::find(v < 0);
        if (!neg.is_empty())
            v.elem(neg).zeros();

        Var.row(i) = v;
    }

    return Var;
}

// Classification infinitesimal jackknife variance
//
// @param PredAll   ntrees x nclass x N cube of per-tree class probabilities
// @param ObsTrack  n_train x ntrees matrix of inbag counts
// @param N         number of test observations
// @param nclass    number of classes
// @return          N x nclass matrix of variance estimates
inline arma::mat compute_cla_ij_variance(const arma::cube& PredAll,
                                          const arma::imat& ObsTrack,
                                          size_t N,
                                          size_t nclass)
{
    // PredAll is (ntrees x nclass x N)
    size_t ntrees = PredAll.n_rows;

    // Clean negative values in ObsTrack
    arma::imat ObsTrack_clean = ObsTrack;
    arma::uvec neg_idx = arma::find(ObsTrack_clean < 0);
    if (!neg_idx.is_empty())
        ObsTrack_clean.elem(neg_idx).zeros();

    // Center inbag counts per observation
    arma::mat counts = arma::conv_to<arma::mat>::from(ObsTrack_clean);
    arma::vec row_means_ot = arma::mean(counts, 1);
    arma::mat N_centered = counts - arma::repmat(row_means_ot, 1, ntrees);

    // NTN = N_centered' * N_centered  (ntrees x ntrees)
    arma::mat NTN = N_centered.t() * N_centered;

    // Monte Carlo bias correction terms
    arma::vec row_means_N = arma::mean(counts, 1);
    arma::vec row_means_sq_N = arma::mean(arma::square(counts), 1);
    double N_var = arma::mean(row_means_sq_N - arma::square(row_means_N));
    size_t n_train = ObsTrack.n_rows;

    arma::mat Var(N, nclass, arma::fill::zeros);

    for (size_t i = 0; i < N; i++)
    {
        // PredAll.slice(i) is ntrees x nclass
        arma::mat Pi = PredAll.slice(i);
        arma::rowvec Pi_mean = arma::mean(Pi, 0);  // 1 x nclass

        // Center predictions per class
        arma::mat pred_centered = Pi - arma::repmat(Pi_mean, ntrees, 1);

        // PNTN = pred_centered' * NTN  (nclass x ntrees)
        arma::mat PNTN = pred_centered.t() * NTN;

        // raw_IJ_c = diag(pred_centered' * NTN * pred_centered) / B^2
        //          = rowSums( (pred_centered * NTN) % pred_centered ) / B^2
        // But doing it per-class:
        arma::rowvec raw_IJ(nclass);
        for (size_t c = 0; c < nclass; c++)
        {
            raw_IJ(c) = arma::dot(pred_centered.col(c), PNTN.row(c).t()) / (ntrees * ntrees);
        }

        // Bias correction per class
        arma::rowvec boot_var = arma::sum(arma::square(pred_centered), 0) / ntrees;
        arma::rowvec bias_corr = (n_train * N_var / ntrees) * boot_var;

        arma::rowvec v = raw_IJ - bias_corr;
        arma::uvec neg = arma::find(v < 0);
        if (!neg.is_empty())
            v.elem(neg).zeros();

        Var.row(i) = v;
    }

    return Var;
}

// Classification jackknife-after-bootstrap variance
//
// @param PredAll   ntrees x nclass x N cube of per-tree class probabilities
// @param ObsTrack  n_train x ntrees matrix of inbag counts
// @param N         number of test observations
// @param nclass    number of classes
// @return          N x nclass matrix of variance estimates
inline arma::mat compute_cla_jack_variance(const arma::cube& PredAll,
                                            const arma::imat& ObsTrack,
                                            size_t N,
                                            size_t nclass)
{
    size_t ntrees = PredAll.n_rows;

    // Clean negative values in ObsTrack
    arma::imat ObsTrack_clean = ObsTrack;
    arma::uvec neg_idx = arma::find(ObsTrack_clean < 0);
    if (!neg_idx.is_empty())
        ObsTrack_clean.elem(neg_idx).zeros();

    // OOB indicator matrix
    arma::umat oob = (ObsTrack_clean == 0);

    // OOB count per training observation
    arma::uvec oob_count = arma::sum(oob, 1);
    arma::uvec valid = arma::find(oob_count > 0);

    if (valid.is_empty())
    {
        Rcpp::warning("No OOB observations found for jackknife variance. Consider increasing ntrees.");
        return arma::mat(N, nclass, arma::fill::value(arma::datum::nan));
    }

    size_t n_valid = valid.n_elem;

    // Forest prediction: average over trees for each observation
    // y_hat is N x nclass
    arma::mat y_hat(N, nclass, arma::fill::zeros);
    for (size_t i = 0; i < N; i++)
        y_hat.row(i) = arma::mean(PredAll.slice(i), 0);

    // Jackknife predictions:
    // For each training obs j, average PredAll over trees where j is OOB
    // jack_pred: N x nclass x n_valid  — compute incrementally
    arma::mat V_J(N, nclass, arma::fill::zeros);

    for (size_t j = 0; j < n_valid; j++)
    {
        size_t idx = valid(j);
        arma::uvec oob_trees = arma::find(oob.row(idx).t());  // column vector of tree indices

        if (oob_trees.is_empty())
            continue;

        // For each test obs i, average over oob_trees
        arma::mat jack_pred_j(N, nclass, arma::fill::zeros);
        for (size_t t = 0; t < oob_trees.n_elem; t++)
        {
            size_t b = oob_trees(t);
            for (size_t i = 0; i < N; i++)
                jack_pred_j.row(i) += PredAll.slice(i).row(b);
        }
        jack_pred_j /= oob_trees.n_elem;

        V_J += arma::square(jack_pred_j - y_hat);
    }
    V_J *= (static_cast<double>(n_valid) - 1.0) / static_cast<double>(n_valid);

    // Bias correction: (e-1) * n_valid / B^2 * sum_b (PredAll[,b,] - y_hat)^2
    arma::mat bias(N, nclass, arma::fill::zeros);
    for (size_t b = 0; b < ntrees; b++)
    {
        for (size_t i = 0; i < N; i++)
        {
            // Need to add row b of slice i minus y_hat row i, squared
            // But PredAll is ntrees x nclass per slice
            bias.row(i) += arma::square(PredAll.slice(i).row(b) - y_hat.row(i));
        }
    }
    bias *= (std::exp(1.0) - 1.0) * static_cast<double>(n_valid) / (ntrees * ntrees);

    // Final variance (floor at 0)
    arma::mat Var = V_J - bias;
    arma::uvec neg_var = arma::find(Var < 0);
    if (!neg_var.is_empty())
        Var.elem(neg_var).zeros();

    return Var;
}

// ============================================================
// Survival variance: per-observation T×T covariance matrix
// ============================================================

// Survival matched-sample variance (per observation)
// pred_i: ntrees × T matrix → T × T covariance matrix
inline arma::mat compute_surv_matched_variance(const arma::mat& pred_i)
{
    size_t B = pred_i.n_rows / 2;
    arma::mat Diff = pred_i.rows(0, B-1) - pred_i.rows(B, 2*B-1);
    arma::mat Vh = Diff.t() * Diff / pred_i.n_rows;
    arma::mat Vs = arma::cov(pred_i, 1);
    arma::mat Cov_raw = Vh - Vs;

    // Project to positive semi-definite
    arma::vec eigval; arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, Cov_raw);
    eigval.elem(arma::find(eigval < 1e-6)).fill(1e-6);
    arma::mat result = eigvec * arma::diagmat(eigval) * eigvec.t();
    return (result + result.t()) / 2;
}

// Survival IJ variance (per observation)
// pred_i: ntrees × T matrix, ObsTrack: n_train × ntrees matrix
inline arma::mat compute_surv_ij_variance(const arma::mat& pred_i,
                                           const arma::imat& ObsTrack)
{
    size_t ntrees = pred_i.n_rows;
    size_t n_train = ObsTrack.n_rows;

    // Clean negative values in ObsTrack
    arma::imat ObsTrack_clean = ObsTrack;
    ObsTrack_clean.elem(arma::find(ObsTrack_clean < 0)).zeros();

    // Center predictions by column (across trees)
    arma::rowvec pred_mean = arma::mean(pred_i, 0);
    arma::mat pred_centered = pred_i - arma::repmat(pred_mean, ntrees, 1);

    // Center ObsTrack per observation
    arma::mat counts = arma::conv_to<arma::mat>::from(ObsTrack_clean);
    arma::vec row_means_ot = arma::mean(counts, 1);
    arma::mat N_centered = counts - arma::repmat(row_means_ot, 1, ntrees);

    // NTN = N_centered' * N_centered  (ntrees × ntrees)
    arma::mat NTN = N_centered.t() * N_centered;

    // V_IJ_raw = pred_centered' * NTN * pred_centered / ntrees²  → T × T
    arma::mat PNTN = pred_centered.t() * NTN;   // T × ntrees
    arma::mat raw_IJ = PNTN * pred_centered / (ntrees * ntrees);  // T × T

    // MC bias correction: (n_train * N_var / ntrees) * cov(pred_centered)
    arma::vec row_means_N = arma::mean(counts, 1);
    arma::vec row_means_sq_N = arma::mean(arma::square(counts), 1);
    double N_var = arma::mean(row_means_sq_N - arma::square(row_means_N));
    arma::mat boot_var = arma::cov(pred_centered, 1);  // T × T
    arma::mat bias_corr = (n_train * N_var / ntrees) * boot_var;

    // Final with positive definite projection
    arma::mat Cov_raw = raw_IJ - bias_corr;
    arma::vec eigval; arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, Cov_raw);
    eigval.elem(arma::find(eigval < 1e-6)).fill(1e-6);
    arma::mat result = eigvec * arma::diagmat(eigval) * eigvec.t();
    return (result + result.t()) / 2;
}

// Survival jackknife variance (per observation)
// pred_i: ntrees × T matrix, ObsTrack: n_train × ntrees matrix
inline arma::mat compute_surv_jack_variance(const arma::mat& pred_i,
                                             const arma::imat& ObsTrack)
{
    size_t ntrees = pred_i.n_rows;
    size_t T = pred_i.n_cols;

    // Clean negative values in ObsTrack
    arma::imat ObsTrack_clean = ObsTrack;
    ObsTrack_clean.elem(arma::find(ObsTrack_clean < 0)).zeros();

    // OOB indicator matrix
    arma::umat oob = (ObsTrack_clean == 0);
    arma::uvec oob_count = arma::sum(oob, 1);
    arma::uvec valid = arma::find(oob_count > 0);

    if (valid.is_empty()) {
        Rcpp::warning("No OOB observations found for jackknife variance.");
        return arma::mat(T, T, arma::fill::value(arma::datum::nan));
    }

    size_t n_valid = valid.n_elem;
    arma::rowvec y_hat = arma::mean(pred_i, 0);  // 1 × T

    arma::mat V_J(T, T, arma::fill::zeros);
    for (size_t j = 0; j < n_valid; j++) {
        arma::uvec oob_trees = arma::find(oob.row(valid(j)).t());
        if (oob_trees.is_empty()) continue;
        arma::rowvec jack_pred = arma::mean(pred_i.rows(oob_trees), 0);  // 1 × T
        arma::vec diff = (jack_pred - y_hat).t();  // T × 1
        V_J += diff * diff.t();  // T × T outer product
    }
    V_J *= (static_cast<double>(n_valid) - 1.0) / static_cast<double>(n_valid);

    // Bias correction: (e-1) * n_valid / ntrees² * Σ_b (pred_i[b,] - y_hat)' * (pred_i[b,] - y_hat)
    arma::mat bias(T, T, arma::fill::zeros);
    for (size_t b = 0; b < ntrees; b++) {
        arma::vec diff = (pred_i.row(b) - y_hat).t();
        bias += diff * diff.t();
    }
    bias *= (std::exp(1.0) - 1.0) * static_cast<double>(n_valid) / (ntrees * ntrees);

    // Final with positive definite projection
    arma::mat Cov_raw = V_J - bias;
    arma::vec eigval; arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, Cov_raw);
    eigval.elem(arma::find(eigval < 1e-6)).fill(1e-6);
    arma::mat result = eigvec * arma::diagmat(eigval) * eigvec.t();
    return (result + result.t()) / 2;
}

} // anonymous namespace

#endif
