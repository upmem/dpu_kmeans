/**
 * @file lloyd_iter.cpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Performs one iteration of the Lloyd K-Means algorithm.
 *
 */
#include <fmt/core.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <utility>

#include "kmeans.hpp"

#ifdef PERF_COUNTER
#include <array>
#endif

extern "C" {
#include <dpu.h>

#include "common.h"
}

namespace {
/**
 * @brief Global array to temporarily store the old centers with padding.
 * Used when the cluster centers buffer isn't 4 aligned.
 */
std::array<int_feature, ASSUMED_NR_CLUSTERS * ASSUMED_NR_FEATURES>
    old_centers_padded;

/**
 * @brief Rounds a number to the next multiple of 4.
 *
 * @param x Number to round.
 * @return size_t Rounded number.
 */
constexpr auto round_to_4(size_t x) -> size_t { return (x + 3U) & ~3U; }

/**
 * @brief Aligns the cluster centers array length to 4 bytes if needed.
 * If the incoming array isn't 4 aligned, it will be copied to a global array.
 *
 * @param old_centers Cluster centers array.
 * @return constexpr auto Pair of the source pointer and the transfer size.
 */
inline auto align_c_clusters(const py::array_t<int_feature> &old_centers) {
  auto xfer_size = static_cast<size_t>(old_centers.nbytes());
  const auto *src_ptr = old_centers.data();

  if (reinterpret_cast<uintptr_t>(src_ptr) % 4 != 0) {
    throw std::runtime_error("old_centers array is not 4 aligned");
  }

  /* If the length of the clusters array isn't 4 aligned, we need
   * to copy it to a new array that will accept padding */
  constexpr int kWramAligment = 4;
  if (xfer_size % kWramAligment != 0) {
    std::copy_n(old_centers.data(), old_centers.size(),
                old_centers_padded.begin());
    xfer_size = round_to_4(xfer_size);
    src_ptr = old_centers_padded.data();
  }

  return std::pair{src_ptr, xfer_size};
}
}  // namespace

void Container::lloyd_iter(const py::array_t<int_feature> &old_centers,
                           py::array_t<int64_t> &centers_psum,
                           py::array_t<int> &centers_pcount) {
  if (!allset_) {
    throw std::runtime_error("No DPUs allocated");
  }
#ifdef PERF_COUNTER
  std::array<uint64_t, HOST_COUNTERS> counters_mean = {};
  std::vector<std::array<uint64_t, HOST_COUNTERS>> counters(p_.ndpu);
#endif

  const auto [src_ptr, xfer_size] = align_c_clusters(old_centers);

  DPU_CHECK(dpu_broadcast_to(allset_.value(), "c_clusters", 0, src_ptr,
                             xfer_size, DPU_XFER_DEFAULT),
            throw std::runtime_error("Failed to broadcast old centers"));

  const auto tic = std::chrono::steady_clock::now();
  //============RUNNING ONE LLOYD ITERATION ON THE DPU==============
  DPU_CHECK(dpu_launch(allset_.value(), DPU_SYNCHRONOUS),
            throw std::runtime_error("Failed to run DPUs"));
  //================================================================
  const auto toc = std::chrono::steady_clock::now();
  p_.time_seconds += std::chrono::duration<double>{toc - tic}.count();

  /* Performance tracking */
#ifdef PERF_COUNTER
  DPU_FOREACH(p_.allset, dpu, each_dpu) {
    DPU_CHECK(dpu_prepare_xfer(dpu, counters[each_dpu].data()));
  }
  DPU_CHECK(dpu_push_xfer(p_.allset, DPU_XFER_FROM_DPU, "host_counters", 0,
                          sizeof(counters[0]), DPU_XFER_DEFAULT));

  for (int icounter = 0; icounter < HOST_COUNTERS; icounter++) {
    int nonzero_dpus = 0;
    for (int idpu = 0; idpu < p_.ndpu; idpu++)
      if (counters[idpu][MAIN_LOOP_CTR] != 0) {
        counters_mean[icounter] += counters[idpu][icounter];
        nonzero_dpus++;
      }
    counters_mean[icounter] /= nonzero_dpus;
  }
  printf("number of %s for this iteration : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[TOTAL_CTR]);
  printf("%s in main loop : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles",
         counters_mean[MAIN_LOOP_CTR]);
  printf("%s in initialization : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[INIT_CTR]);
  printf("%s in critical loop arithmetic : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles",
         counters_mean[CRITLOOP_ARITH_CTR]);
  printf("%s in reduction arithmetic + implicit access : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles",
         counters_mean[REDUCE_ARITH_CTR]);
  printf("%s in reduction loop : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles",
         counters_mean[REDUCE_LOOP_CTR]);
  printf("%s in dispatch function : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles",
         counters_mean[DISPATCH_CTR]);
  printf("%s in mutexed implicit access : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[MUTEX_CTR]);

  printf("\ntotal %s in arithmetic : %ld\n",
         (PERF_COUNTER) ? "instructions" : "cycles",
         counters_mean[CRITLOOP_ARITH_CTR] + counters_mean[REDUCE_ARITH_CTR]);
  printf("percent %s in arithmetic : %.2f%%\n",
         (PERF_COUNTER) ? "instructions" : "cycles",
         100.0 *
             (float)(counters_mean[CRITLOOP_ARITH_CTR] +
                     counters_mean[REDUCE_ARITH_CTR]) /
             counters_mean[TOTAL_CTR]);
  printf("\n");
#endif

  /* copy back membership count per dpu (device to host) */
  dpu_set_t dpu{};
  uint32_t each_dpu = 0;
  DPU_FOREACH(allset_.value(), dpu, each_dpu) {
    DPU_CHECK(
        // TODO: direct access
        dpu_prepare_xfer(dpu, centers_pcount.mutable_data(each_dpu)),
        throw std::runtime_error("Failed to prepare transfer"));
  }
  auto nr_clusters = centers_pcount.shape(1);
  if (nr_clusters > ASSUMED_NR_CLUSTERS) {
    throw std::length_error(
        fmt::format("Too many clusters for one DPU : {}", nr_clusters));
  }
  DPU_CHECK(dpu_push_xfer(
                allset_.value(), DPU_XFER_FROM_DPU, "centers_count_mram", 0,
                static_cast<size_t>(centers_pcount.itemsize() * nr_clusters),
                DPU_XFER_DEFAULT),
            throw std::runtime_error("Failed to push transfer"));

  /* copy back centroids partial sums (device to host) */
  DPU_FOREACH(allset_.value(), dpu, each_dpu) {
    DPU_CHECK(dpu_prepare_xfer(dpu, centers_psum.mutable_data(each_dpu, 0, 0)),
              throw std::runtime_error("Failed to prepare transfer"));
  }
  auto nr_clusters_x_nr_features =
      centers_psum.shape(1) * centers_psum.shape(2);
  if (nr_clusters_x_nr_features > ASSUMED_NR_CLUSTERS * ASSUMED_NR_FEATURES) {
    throw std::length_error(
        fmt::format("Too many clusters x features for one DPU : {}",
                    nr_clusters_x_nr_features));
  }
  DPU_CHECK(
      dpu_push_xfer(allset_.value(), DPU_XFER_FROM_DPU, "centers_sum_mram", 0,
                    static_cast<size_t>(centers_psum.itemsize() *
                                        nr_clusters_x_nr_features),
                    DPU_XFER_DEFAULT),
      throw std::runtime_error("Failed to push transfer"));

  /* averaging the new centers and summing the centers count
   * has been moved to the python code */
}

auto Container::get_inertia() -> int64_t {
  if (!allset_) {
    throw std::runtime_error("No DPUs allocated");
  }
  auto tic = std::chrono::steady_clock::now();
  /* Copy back inertia (device to host) */
  dpu_set_t dpu{};
  uint32_t each_dpu = 0;
  DPU_FOREACH(allset_.value(), dpu, each_dpu) {
    DPU_CHECK(dpu_prepare_xfer(dpu, &(inertia_per_dpu_[each_dpu])),
              throw std::runtime_error("Failed to prepare transfer"));
  }
  DPU_CHECK(dpu_push_xfer(allset_.value(), DPU_XFER_FROM_DPU, "inertia", 0,
                          sizeof(inertia_per_dpu_[0]), DPU_XFER_DEFAULT),
            throw std::runtime_error("Failed to push transfer"));

  auto toc = std::chrono::steady_clock::now();
  p_.cpu_pim_time += std::chrono::duration<double>{toc - tic}.count();

  return std::accumulate(inertia_per_dpu_.cbegin(), inertia_per_dpu_.cend(),
                         0LL);
}
