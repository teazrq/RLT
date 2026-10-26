#ifndef RLT_FOREST_BUILD_GUARDS_H
#define RLT_FOREST_BUILD_GUARDS_H

#include <exception>
#include <stdexcept>
#include <vector>

// Each iteration writes its own exception slot. Rcpp translates the rethrow
// only after the outermost forest has left its OpenMP region.
inline void rethrow_tree_build_errors(const std::vector<std::exception_ptr>& errors)
{
  for (const auto& error : errors)
    if (error)
      std::rethrow_exception(error);
}

inline void check_sampled_tree_weight(const arma::vec& weights,
                                      const arma::uvec& inbag_id)
{
  if (!(arma::sum(weights(inbag_id)) > 0))
    throw std::runtime_error("A sampled tree has zero total observation weight.");
}

inline bool positive_child_weights(const arma::vec& weights,
                                    const arma::uvec& left_id,
                                    const arma::uvec& right_id)
{
  return arma::sum(weights(left_id)) > 0 && arma::sum(weights(right_id)) > 0;
}

inline double survival_node_weight(double weight, double total_weight,
                                    double mean_weight, size_t node_size)
{
  // Keep the existing arithmetic unless a positive mean underflows to zero.
  return mean_weight > 0 ? weight / mean_weight :
    (weight / total_weight) * node_size;
}

#endif
