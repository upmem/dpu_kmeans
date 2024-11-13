/**
 * @file dimm_manager.cpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Nanny functions for the DPUs.
 *
 */

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../kmeans.hpp"

extern "C" {
#include <dpu.h>
#include <dpu_types.h>
}

/**
 * @brief Allocates all DPUs
 *
 * @param p Algorithm parameters.
 */
void Container::allocate() {
  if (p_.ndpu == 0U) {
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, nullptr, &p_.allset));
  } else {
    DPU_ASSERT(dpu_alloc(p_.ndpu, nullptr, &p_.allset));
  }
  p_.allocated = true;
  DPU_ASSERT(dpu_get_nr_dpus(p_.allset, &p_.ndpu));
  inertia_per_dpu_.resize(p_.ndpu);
}

/**
 * @brief Frees the DPUs.
 *
 * @param p Algorithm parameters.
 */
void Container::free_dpus() {
  if (p_.allocated) {
    DPU_ASSERT(dpu_free(p_.allset));
    p_.allocated = false;
  }
}

/**
 * @brief Loads a binary in the DPUs.
 *
 * @param p Algorithm parameters.
 * @param DPU_BINARY path to the binary
 */
void Container::load_kernel(const std::filesystem::path &binary_path) {
  DPU_ASSERT(dpu_load(p_.allset, binary_path.c_str(), nullptr));
}

void Container::broadcast_number_of_clusters() const {
  unsigned int nclusters_short = p_.nclusters;
  DPU_ASSERT(dpu_broadcast_to(p_.allset, "nclusters_host", 0, &nclusters_short,
                              sizeof(nclusters_short), DPU_XFER_DEFAULT));
}

void Container::populate_dpus(const py::array_t<int_feature> &py_features) {
  auto features = py_features.unchecked<2>();

  nreal_points_.resize(p_.ndpu);
  int64_t padding_points =
      p_.npadded - p_.npoints; /* number of padding points */

  const auto tic = std::chrono::steady_clock::now();

  dpu_set_t dpu{};
  uint32_t each_dpu = 0;
  int64_t next = 0;
  DPU_FOREACH(p_.allset, dpu, each_dpu) {
    int64_t current = next;
    /* The C API takes a non-const pointer but does not modify the data */
    DPU_ASSERT(dpu_prepare_xfer(
        dpu, const_cast<int_feature *>(features.data(next, 0))));
    padding_points -= p_.npointperdpu;
    next = std::max(0L, -padding_points);

    int64_t nreal_points_dpu = next - current;
    if (nreal_points_dpu > std::numeric_limits<int>::max()) {
      throw std::length_error(
          fmt::format("Too many points for one DPU : {}", nreal_points_dpu));
    }
    nreal_points_[each_dpu] = static_cast<int>(nreal_points_dpu);
  }
  DPU_ASSERT(dpu_push_xfer(p_.allset, DPU_XFER_TO_DPU, "t_features", 0,
                           p_.npointperdpu * p_.nfeatures * sizeof(int_feature),
                           DPU_XFER_DEFAULT));

  DPU_FOREACH(p_.allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &nreal_points_[each_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(p_.allset, DPU_XFER_TO_DPU, "npoints", 0,
                           sizeof(int), DPU_XFER_DEFAULT));

  const auto toc = std::chrono::steady_clock::now();
  p_.cpu_pim_time = std::chrono::duration<double>{toc - tic}.count();
}

void Container::load_nclusters(int nclusters) {
  p_.nclusters = nclusters;

  broadcast_number_of_clusters();
}

/**
 * @brief Computes the appropriate task size for DPU tasklets.
 *
 * @param p Algorithm parameters.
 * @return The task size in bytes.
 */
[[nodiscard]] constexpr auto Container::get_task_size() const -> int {
  /* how many points we can fit in w_features */
  int max_task_size =
      (WRAM_FEATURES_SIZE / static_cast<int>(sizeof(int_feature))) /
      p_.nfeatures;

  /* number of tasks as the smallest multiple of NR_TASKLETS higher than
   * npointperdu / max_task_size */
  int ntasks =
      static_cast<int>((p_.npointperdpu + max_task_size - 1) / max_task_size);
  ntasks = ((ntasks + NR_TASKLETS - 1) / NR_TASKLETS) * NR_TASKLETS;

  /* task size in points should fit in an int */
  int64_t task_size_in_points_64 = (p_.npointperdpu + ntasks - 1) / ntasks;
  if (task_size_in_points_64 > std::numeric_limits<int>::max()) {
    throw std::length_error(fmt::format("task size in points is too large: {}",
                                        task_size_in_points_64));
  }
  /* task size has to be at least 1 and at most max_task_size */
  int task_size_in_points =
      std::clamp(static_cast<int>(task_size_in_points_64), 1, max_task_size);

  int task_size_in_features = task_size_in_points * p_.nfeatures;
  int task_size_in_bytes =
      task_size_in_features * static_cast<int>(sizeof(int_feature));

  /* task size in bytes must be a multiple of 8 for DMA alignment and also a
   * multiple of number of features x byte size of integers */
  int lcm = std::lcm(static_cast<int>(sizeof(int_feature)) * p_.nfeatures, 8);
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

void Container::broadcast_parameters() {
  /* parameters to calculate once here and send to the DPUs. */

  /* compute the iteration variables for the DPUs */
  int task_size_in_bytes = get_task_size();

  /* realign task size in features and points */
  int task_size_in_features =
      task_size_in_bytes / static_cast<int>(sizeof(int_feature));
  int task_size_in_points = task_size_in_features / p_.nfeatures;

  /* send computation parameters to the DPUs */
  DPU_ASSERT(dpu_broadcast_to(p_.allset, "nfeatures_host", 0, &p_.nfeatures,
                              sizeof(p_.nfeatures), DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_broadcast_to(p_.allset, "task_size_in_points_host", 0,
                              &task_size_in_points, sizeof(task_size_in_points),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p_.allset, "task_size_in_bytes_host", 0,
                              &task_size_in_bytes, sizeof(task_size_in_bytes),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p_.allset, "task_size_in_features_host", 0,
                              &task_size_in_features,
                              sizeof(task_size_in_features), DPU_XFER_DEFAULT));
}

void Container::transfer_data(const py::array_t<int_feature> &data_int) {
  populate_dpus(data_int);
  broadcast_parameters();
}

void Container::load_array_data(const py::array_t<int_feature> &data_int,
                                int64_t npoints, int nfeatures) {
  p_.npoints = npoints;
  p_.nfeatures = nfeatures;
  p_.npadded = ((p_.npoints + 8 * p_.ndpu - 1) / (8 * p_.ndpu)) * 8 * p_.ndpu;
  p_.npointperdpu = p_.npadded / p_.ndpu;

  transfer_data(data_int);
}
