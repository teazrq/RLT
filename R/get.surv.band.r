#' @title get.surv.band
#' @description Calculate the survival function (two-sided) confidence band from a RLT survival prediction.
#' @importFrom stats predict
#' @param x     A RLT prediction object. Must be from a forest with var.mode != "none".
#' @param i     Observation number in the prediction. Default to calculate all (i = 0).
#' @param alpha alpha level for interval (alpha/2, 1 - alpha/2).
#' @param approach Confidence band approach:
#'        - naive: marsd = sqrt(diag(Cov)), MC band using full covariance.
#'        - smoothed: GAM-smoothed rank-K covariance + eigenvalue-ratio weighted residual correction.
#' @param nsim  Number of simulations for estimating the Monte Carlo critical value.
#' @param k_rank Rank truncation K used for the smooth low-rank covariance AND GAM basis size.
#' @param k_mode Rank selection mode: "fixed" (use k_rank) or "proportion" (auto-select by eigenvalue cumulative ratio).
#' @param k_prop Proportion threshold (0,1] for cumulative eigenvalue ratio when k_mode = "proportion".
#' @param ...   Further arguments (currently not used).
#' @return An object of class \code{c("RLT", "band", "surv")} with components: \code{lower} (lower bound), \code{upper} (upper bound), and \code{timepoints} (evaluation grid). If \code{i = 0}, a list of such objects for all observations.
#' @export
#' @examples
#' \donttest{
#'   set.seed(42)
#'   n <- 100
#'   x <- matrix(rnorm(n * 5), ncol = 5)
#'   y <- rexp(n, rate = exp(rowSums(x[, 1:2])))
#'   censor <- rbinom(n, 1, 0.7)
#'   fit <- RLT(x, y, censor = censor, model = "survival",
#'              ntrees = 200, var.mode = "matched")
#'   pred <- predict(fit, testx = x[1:3, ], var.est = TRUE)
#'   band <- get.surv.band(pred, i = 1, alpha = 0.05)
#' }
get.surv.band <- function(x,
                          i        = 0,
                          alpha    = 0.05,
                          approach = "naive",
                          nsim     = 5000,
                          k_rank   = 10,
                          k_mode   = c("fixed", "proportion"),
                          k_prop   = 0.99,
                          ...)
{
  if (any(class(x)[1:3] != c("RLT", "pred", "surv")))
    stop("Not an RLT survival prediction object.")

  if (is.null(x$Cov))
    stop("Not an RLT object fitted with var.mode")

  all.approach <- c("naive", "smoothed")
  if (match(approach, all.approach, nomatch = 0) == 0)
    stop("approach not available")

  N  <- dim(x$Cov)[3]
  p  <- dim(x$Cov)[2]
  nt <- nrow(x$Cov[,,1])

  if (p != nt)
    stop("Internal dimension mismatch (p vs nt)")

  if (i == 0) {
    allid <- 1:N
  } else {
    if (i < 0 | i > N)
      stop(paste("Observation", i, "does not exist"))
    allid <- i
  }

  if (any(alpha < 0) | any(alpha > 0.5))
    stop("alpha not valid")

  k_mode <- match.arg(k_mode)
  if (k_mode == "proportion" && (k_prop <= 0 || k_prop > 1))
    stop("k_prop must be in (0, 1] when k_mode = 'proportion'")

  SurvBand <- list()
  eps <- .Machine$double.eps

  ## ---------------- helpers ----------------

  nearest_psd <- function(M) {
    eg   <- eigen(M, symmetric = TRUE)
    vals <- eg$values
    vals[vals < 0] <- 0
    Mp <- eg$vectors %*% (vals * t(eg$vectors))
    (Mp + t(Mp)) / 2
  }

  gam_smooth_cov <- function(G, k_rank) {
    if (!requireNamespace("mgcv", quietly = TRUE))
      stop("Package 'mgcv' needed for smoothing. Please install it.", call. = FALSE)

    k_safe  <- max(k_rank, 3)
    k_basis <- c(k_safe, k_safe)

    n  <- nrow(G)
    ij <- expand.grid(i = 1:n, j = 1:n)
    z  <- as.vector(G[ij$i + (ij$j - 1L) * n])

    fit <- mgcv::gam(z ~ te(i, j, k = k_basis), data = ij)
    Gs  <- matrix(predict(fit, newdata = ij), n, n)
    (Gs + t(Gs)) / 2
  }

  rank_k_psd <- function(Sigma_psd, k_rank) {
    p <- nrow(Sigma_psd)
    k <- min(max(k_rank, 0), p)
    if (k == 0) return(matrix(0, p, p))

    eg   <- eigen(Sigma_psd, symmetric = TRUE)
    lam  <- pmax(eg$values, 0)
    U    <- eg$vectors

    U_k  <- U[, seq_len(k), drop = FALSE]
    lamk <- lam[seq_len(k)]
    SigK <- U_k %*% (lamk * t(U_k))
    (SigK + t(SigK)) / 2
  }

  select_rank_by_eig <- function(Sigma_psd, threshold = 0.99, eps = .Machine$double.eps) {
    eg   <- eigen(Sigma_psd, symmetric = TRUE)
    vals <- pmax(eg$values, 0)
    total <- sum(vals)
    if (total <= eps)
      return(list(k = 0, var_explained = 0, eigen = eg, vals = vals, cum_ratio = rep(0, length(vals))))
    cum_ratio <- cumsum(vals) / total
    k <- which(cum_ratio >= threshold)[1]
    if (is.na(k)) k <- length(vals)
    list(k = k, var_explained = cum_ratio[k], eigen = eg, vals = vals, cum_ratio = cum_ratio)
  }

  pos_spectrum_psd <- function(M) {
    eg  <- eigen(M, symmetric = TRUE)
    mu  <- eg$values
    U   <- eg$vectors
    pos <- which(mu > 0)
    if (length(pos) == 0) return(matrix(0, nrow(M), ncol(M)))
    Upos  <- U[, pos, drop = FALSE]
    mupos <- mu[pos]
    Mpos  <- Upos %*% (mupos * t(Upos))
    (Mpos + t(Mpos)) / 2
  }

  ## ---------------- main loop ----------------

  band_idx <- 0L

  for (idx in allid) {
    band_idx <- band_idx + 1L

    Sigma_raw <- x$Cov[,,idx]
    raw_sd <- sqrt(pmax(diag(Sigma_raw), eps))
    eig_info <- NULL

    if (approach == "naive") {

      marsd <- sqrt(pmax(diag(Sigma_raw), eps))
      bandk <- mc_band(marsd, Sigma_raw, alpha, nsim)

      diag_sd_info <- data.frame(
        timepoint = x$timepoints,
        raw_sd    = raw_sd,
        final_sd  = marsd
      )
      var_explained_info <- list(
        var_explained = 1.0,
        k_rank        = NA,
        total_var     = sum(diag(Sigma_raw)),
        lowrank_var   = sum(diag(Sigma_raw))
      )
    } else if (approach == "smoothed") {

      ## 1) choose k_used
      if (k_mode == "proportion") {
        sel    <- select_rank_by_eig(nearest_psd(Sigma_raw), threshold = k_prop, eps = eps)
        k_used <- sel$k
        eig_info <- list(
          vals = sel$vals, cum_ratio = sel$cum_ratio,
          k_used = k_used, k_prop = k_prop,
          var_explained = sel$var_explained
        )
      } else {
        k_used <- k_rank
      }

      ## 2) smooth full covariance via GAM + PSD
      Sigma_s   <- gam_smooth_cov(Sigma_raw, k_rank = k_used)
      Sigma_psd <- nearest_psd(Sigma_s)

      ## 3) rank-K main part
      Sigma_smoothK <- if (k_used == 0) matrix(0, nrow(Sigma_psd), ncol(Sigma_psd))
                       else rank_k_psd(Sigma_psd, k_rank = k_used)

      smoothK_sd <- sqrt(pmax(diag(Sigma_smoothK), eps))

      ## 4) residual PSD (positive spectrum of Sigma_raw - Sigma_smoothK)
      R     <- (Sigma_raw - Sigma_smoothK + t(Sigma_raw - Sigma_smoothK)) / 2
      R_pos <- pos_spectrum_psd(R)

      ## 5) constant SD from residual
      sdB <- sqrt(pmax(max(diag(R_pos)), 0))

      ## 6) eigenvalue-ratio weights
      tr_smoothK <- sum(pmax(diag(Sigma_smoothK), 0))
      tr_R_pos   <- sum(pmax(diag(R_pos), 0))
      tr_total   <- tr_smoothK + tr_R_pos

      if (tr_total > eps) {
        w_k_eig <- tr_smoothK / tr_total
        w_r_eig <- tr_R_pos / tr_total
      } else {
        w_k_eig <- 1.0
        w_r_eig <- 0.0
      }

      ## 7) final SD = sqrt(w_k * smoothK_sd^2 + w_r * resid_sd^2)
      marsd <- sqrt(pmax(w_k_eig * smoothK_sd^2 + w_r_eig * sdB^2, eps))
      bandk <- mc_band(marsd, Sigma_raw, alpha, nsim)

      ## diagnostics
      total_raw_var  <- sum(diag(Sigma_raw))
      smooth_trace   <- sum(diag(Sigma_smoothK))
      info_retention <- smooth_trace / total_raw_var
      ratio_max_mean <- max(diag(R_pos)) / mean(diag(R_pos) + eps)
      ratio_vec      <- marsd / raw_sd
      min_ratio      <- min(ratio_vec)
      argmin         <- which.min(ratio_vec)
      t_min          <- x$timepoints[argmin]

      diag_sd_info <- data.frame(
        timepoint  = x$timepoints,
        raw_sd     = raw_sd,
        smoothK_sd = smoothK_sd,
        final_sd   = marsd,
        resid_sd   = rep(sdB, length(x$timepoints))
      )
      var_explained_info <- list(
        var_explained = info_retention,
        k_rank        = k_used,
        total_var     = total_raw_var,
        lowrank_var   = smooth_trace,
        resid_var     = total_raw_var - smooth_trace
      )
    }

    # Band construction: exp(-CHF ± bandk)
    SurvBand[[band_idx]] <- list(
      lower        = exp(-x$CHF[idx, ] - bandk),
      upper        = exp(-x$CHF[idx, ] + bandk),
      diagnostics  = list(
        diag_sd       = diag_sd_info,
        var_explained = var_explained_info,
        eig           = eig_info
      )
    )
  }

  names(SurvBand) <- paste("Subject", allid, sep = "")
  SurvBand[[length(allid) + 1]] <- x$timepoints
  names(SurvBand)[length(SurvBand)] <- "timepoints"
  class(SurvBand) <- c("RLT", "band", "surv")
  return(SurvBand)
}
