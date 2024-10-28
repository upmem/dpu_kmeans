/**
 * @file dimm_manager.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Nanny functions for the DPUs in C++.
 *
 */

#include "dimm_manager.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

extern "C" {
#include <dpu.h>
}

/**
 * @brief Allocates all DPUs
 *
 * @param p Algorithm parameters.
 */
void allocate_dpus(kmeans_params &p) {
  if (!p.ndpu) {
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, nullptr, &p.allset));
  } else {
    DPU_ASSERT(dpu_alloc(p.ndpu, nullptr, &p.allset));
  }
  DPU_ASSERT(dpu_get_nr_dpus(p.allset, &p.ndpu));
}

/**
 * @brief Frees the DPUs.
 *
 * @param p Algorithm parameters.
 */
void free_dpus(const kmeans_params &p) { DPU_ASSERT(dpu_free(p.allset)); }

/**
 * @brief Loads a binary in the DPUs.
 *
 * @param p Algorithm parameters.
 * @param DPU_BINARY path to the binary
 */
void load_kernel_internal(const kmeans_params &p,
                          const std::filesystem::path &binary_path) {
  DPU_ASSERT(dpu_load(p.allset, binary_path.c_str(), nullptr));
}

/**
 * @brief Broadcast current number of clusters to the DPUs
 *
 * @param p Algorithm parameters.
 * @param nclusters Number of clusters.
 */
void broadcast_number_of_clusters(const kmeans_params &p, int nclusters) {
  /* inform DPUs of the current number of clusters */
  unsigned int nclusters_short = nclusters;
  DPU_ASSERT(dpu_broadcast_to(p.allset, "nclusters_host", 0, &nclusters_short,
                              sizeof(nclusters_short), DPU_XFER_DEFAULT));
}

/**
 * @brief Fills the DPUs with their assigned points.
 */
void populate_dpus(kmeans_params &p, /**< Algorithm parameters */
                   const py::array_t<int_feature>
                       &py_features) /**< array: [npoints][nfeatures] */
{
  auto features = py_features.unchecked<2>();

  /* Iteration variables for the DPUs. */
  dpu_set_t dpu{};
  uint32_t each_dpu = 0;

  std::vector<int> nreal_points(
      p.ndpu); /* number of real data points on each dpu */
  int64_t padding_points = p.npadded - p.npoints; /* number of padding points */

  const auto tic = std::chrono::steady_clock::now();

  int64_t next = 0;
  DPU_FOREACH(p.allset, dpu, each_dpu) {
    int64_t current = next;
    /* The C API takes a non-const pointer but does not modify the data */
    DPU_ASSERT(dpu_prepare_xfer(
        dpu, const_cast<int_feature *>(features.data(next, 0))));
    padding_points -= p.npointperdpu;
    next = std::max(0L, -padding_points);

    int64_t nreal_points_dpu = next - current;
    if (nreal_points_dpu > std::numeric_limits<int>::max()) {
      throw std::length_error(
          fmt::format("Too many points for one DPU : {}", nreal_points_dpu));
    }
    nreal_points[each_dpu] = static_cast<int>(nreal_points_dpu);
  }
  DPU_ASSERT(dpu_push_xfer(p.allset, DPU_XFER_TO_DPU, "t_features", 0,
                           p.npointperdpu * p.nfeatures * sizeof(int_feature),
                           DPU_XFER_DEFAULT));

  DPU_FOREACH(p.allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &nreal_points[each_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(p.allset, DPU_XFER_TO_DPU, "npoints", 0, sizeof(int),
                           DPU_XFER_DEFAULT));

  const auto toc = std::chrono::steady_clock::now();
  p.cpu_pim_time = std::chrono::duration<double>{toc - tic}.count();
}

/**
 * @brief Computes the appropriate task size for DPU tasklets.
 *
 * @param p Algorithm parameters.
 * @return The task size in bytes.
 */
static constexpr auto get_task_size(const kmeans_params &p) -> int {
  /* how many points we can fit in w_features */
  int max_task_size =
      (WRAM_FEATURES_SIZE / static_cast<int>(sizeof(int_feature))) /
      p.nfeatures;

  /* number of tasks as the smallest multiple of NR_TASKLETS higher than
   * npointperdu / max_task_size */
  int ntasks =
      static_cast<int>((p.npointperdpu + max_task_size - 1) / max_task_size);
  ntasks = ((ntasks + NR_TASKLETS - 1) / NR_TASKLETS) * NR_TASKLETS;

  /* task size in points should fit in an int */
  int64_t task_size_in_points_64 = (p.npointperdpu + ntasks - 1) / ntasks;
  if (task_size_in_points_64 > std::numeric_limits<int>::max()) {
    throw std::length_error(fmt::format("task size in points is too large: {}",
                                        task_size_in_points_64));
  }
  /* task size has to be at least 1 and at most max_task_size */
  int task_size_in_points =
      std::clamp(static_cast<int>(task_size_in_points_64), 1, max_task_size);

  int task_size_in_features = task_size_in_points * p.nfeatures;
  int task_size_in_bytes =
      task_size_in_features * static_cast<int>(sizeof(int_feature));

  /* task size in bytes must be a multiple of 8 for DMA alignment and also a
   * multiple of number of features x byte size of integers */
  int lcm = std::lcm(static_cast<int>(sizeof(int_feature)) * p.nfeatures, 8);
  task_size_in_bytes = task_size_in_bytes / lcm * lcm;
  if (task_size_in_bytes > WRAM_FEATURES_SIZE) {
    throw std::length_error(
        fmt::format("tasks will not fit in WRAM, task size in bytes: {}",
                    task_size_in_bytes));
  }
  /* minimal size */
  task_size_in_bytes = std::max(task_size_in_bytes, lcm);

  return task_size_in_bytes;
}

/**
 * @brief Broadcasts iteration parameters to the DPUs.
 *
 * @param p Algorithm parameters.
 */
void broadcast_parameters(const kmeans_params &p) {
  /* parameters to calculate once here and send to the DPUs. */

  /* compute the iteration variables for the DPUs */
  int task_size_in_bytes = get_task_size(p);

  /* realign task size in features and points */
  int task_size_in_features =
      task_size_in_bytes / static_cast<int>(sizeof(int_feature));
  int task_size_in_points = task_size_in_features / p.nfeatures;

  /* send computation parameters to the DPUs */
  DPU_ASSERT(dpu_broadcast_to(p.allset, "nfeatures_host", 0, &p.nfeatures,
                              sizeof(p.nfeatures), DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_broadcast_to(p.allset, "task_size_in_points_host", 0,
                              &task_size_in_points, sizeof(task_size_in_points),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p.allset, "task_size_in_bytes_host", 0,
                              &task_size_in_bytes, sizeof(task_size_in_bytes),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p.allset, "task_size_in_features_host", 0,
                              &task_size_in_features,
                              sizeof(task_size_in_features), DPU_XFER_DEFAULT));
}
