/**
 * @file dimm_manager.cpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Nanny functions for the DPUs.
 *
 */

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "kmeans.hpp"

extern "C" {
#include <dpu.h>
#include <dpu_types.h>
}

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

void Container::free_dpus() {
  if (p_.allocated) {
    DPU_ASSERT(dpu_free(p_.allset));
    p_.allocated = false;
  }
}

void Container::load_kernel(const std::filesystem::path &binary_path) {
  DPU_ASSERT(dpu_load(p_.allset, binary_path.c_str(), nullptr));
}

/**
 * @brief Utility function to check the cast of a value.
 *
 * @tparam T type to cast to
 * @tparam U type of the value to cast
 * @param name name of the variable to be cast
 * @param value value to cast
 * @return T the casted value
 */
template <typename T, typename U>
static constexpr auto checked_cast(std::string_view name, U value) -> T {
  if (value > std::numeric_limits<T>::max()) {
    throw std::overflow_error(fmt::format("{} is too large: {} (max {})", name,
                                          value,
                                          std::numeric_limits<T>::max()));
  }
  return static_cast<T>(value);
}

/**
 * @brief utility macro to create a name-value pair for checked_cast
 *
 */
#define VN(var) #var, var

void Container::broadcast_number_of_clusters() const {
  checked_cast<uint8_t>(VN(p_.nclusters));
  DPU_ASSERT(dpu_broadcast_to(p_.allset, "nclusters_host", 0, &p_.nclusters,
                              sizeof(p_.nclusters), DPU_XFER_DEFAULT));
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
    nreal_points_[each_dpu] = checked_cast<int>(VN(nreal_points_dpu));
  }
  auto features_count_per_dpu = p_.npointperdpu * p_.nfeatures;
  if (features_count_per_dpu > MAX_FEATURE_DPU) {
    throw std::length_error(fmt::format("Too many features for one DPU : {}",
                                        features_count_per_dpu));
  }
  DPU_ASSERT(dpu_push_xfer(
      p_.allset, DPU_XFER_TO_DPU, "t_features", 0,
      static_cast<size_t>(features_count_per_dpu) * sizeof(int_feature),
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
    throw std::overflow_error(fmt::format(
        "task size in points is too large: {}", task_size_in_points_64));
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

  /* validate variables width for the DPUs */
  task_parameters params_host{checked_cast<uint8_t>(VN(p_.nfeatures)),
                              checked_cast<uint8_t>(VN(task_size_in_points)),
                              checked_cast<uint16_t>(VN(task_size_in_features)),
                              checked_cast<uint16_t>(VN(task_size_in_bytes))};

  /* send computation parameters to the DPUs */
  DPU_ASSERT(dpu_broadcast_to(p_.allset, "p_h", 0, &params_host,
                              sizeof(task_parameters), DPU_XFER_DEFAULT));
}

#undef VN

void Container::transfer_data(const py::array_t<int_feature> &data_int) {
  populate_dpus(data_int);
  broadcast_parameters();
}

void Container::load_array_data(const py::array_t<int_feature> &data_int,
                                int64_t npoints, int nfeatures) {
  p_.npoints = npoints;
  p_.nfeatures = nfeatures;
  auto alignment = 8 * p_.ndpu;
  p_.npadded = ((p_.npoints + alignment - 1) / alignment) * alignment;
  p_.npointperdpu = p_.npadded / p_.ndpu;

  transfer_data(data_int);
}
