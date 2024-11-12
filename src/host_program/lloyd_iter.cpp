/**
 * @file lloyd_iter.cpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Performs one iteration of the Lloyd K-Means algorithm.
 *
 */

#include "lloyd_iter.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <numeric>

#ifdef PERF_COUNTER
#include <array>
#endif

extern "C" {
#include "../kmeans.h"
}

/**
 * @brief Performs one iteration of the Lloyd algorithm on DPUs and gets the
 * results.
 *
 * @param p Algorithm parameters.
 * @param old_centers [in] Discretized current centroids coordinates.
 * @param centers_psum [out] Sum of points coordinates per cluster per dpu.
 * @param centers_pcount [out] Count of elements in each cluster per dpu.
 */
void lloyd_iter(kmeans_params &p, const py::array_t<int_feature> &old_centers,
                py::array_t<int64_t> &centers_psum,
                py::array_t<int> &centers_pcount) {
  dpu_set_t dpu{};       /* Iteration variable for the DPUs. */
  uint32_t each_dpu = 0; /* Iteration variable for the DPUs. */

#ifdef PERF_COUNTER
  std::array<uint64_t, HOST_COUNTERS> counters_mean = {};
  std::vector<std::array<uint64_t, HOST_COUNTERS>> counters(p.ndpu);
#endif

  DPU_ASSERT(dpu_broadcast_to(
      p.allset, "c_clusters", 0, old_centers.data(0, 0),
      static_cast<size_t>(p.nclusters) * p.nfeatures * sizeof(int_feature),
      DPU_XFER_DEFAULT));

  const auto tic = std::chrono::steady_clock::now();
  //============RUNNING ONE LLOYD ITERATION ON THE DPU==============
  DPU_ASSERT(dpu_launch(p.allset, DPU_SYNCHRONOUS));
  //================================================================
  const auto toc = std::chrono::steady_clock::now();
  p.time_seconds += std::chrono::duration<double>{toc - tic}.count();

  /* Performance tracking */
#ifdef PERF_COUNTER
  DPU_FOREACH(p.allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, counters[each_dpu].data()));
  }
  DPU_ASSERT(dpu_push_xfer(p.allset, DPU_XFER_FROM_DPU, "host_counters", 0,
                           sizeof(counters[0]), DPU_XFER_DEFAULT));

  for (int icounter = 0; icounter < HOST_COUNTERS; icounter++) {
    int nonzero_dpus = 0;
    for (int idpu = 0; idpu < p.ndpu; idpu++)
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
  DPU_FOREACH(p.allset, dpu, each_dpu) {
    DPU_ASSERT(
        // TODO: direct access
        dpu_prepare_xfer(dpu, centers_pcount.mutable_data(each_dpu)));
  }
  DPU_ASSERT(dpu_push_xfer(p.allset, DPU_XFER_FROM_DPU, "centers_count_mram", 0,
                           centers_pcount.itemsize() * centers_pcount.shape(1),
                           DPU_XFER_DEFAULT));

  /* copy back centroids partial sums (device to host) */
  DPU_FOREACH(p.allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, centers_psum.mutable_data(each_dpu)));
  }
  DPU_ASSERT(dpu_push_xfer(
      p.allset, DPU_XFER_FROM_DPU, "centers_sum_mram", 0,
      static_cast<long>(p.nfeatures) * p.nclusters * sizeof(int64_t),
      DPU_XFER_DEFAULT));

  /* averaging the new centers and summing the centers count
   * has been moved to the python code */
}

/**
 * @brief Performs one E step of the Lloyd algorithm on DPUs and gets the
 * inertia only.
 *
 * @param p Algorithm parameters.
 * @param old_centers [in] Discretized current centroids coordinates.
 * @param inertia_psum [out] Buffer to read inertia per DPU.
 */
auto lloyd_iter_with_inertia(
    kmeans_params &p, /**< Algorithm parameters. */
    const py::array_t<int_feature>
        &old_centers, /**< [in] Discretized current centroids coordinates. */
    std::vector<int64_t> &inertia_psum /**< Buffer to read inertia per DPU. */
    ) -> int64_t {
  dpu_set_t dpu{};       /* Iteration variable for the DPUs. */
  uint32_t each_dpu = 0; /* Iteration variable for the DPUs. */

  int compute_inertia = 1;

  DPU_ASSERT(dpu_broadcast_to(p.allset, "c_clusters", 0, old_centers.ptr(),
                              old_centers.nbytes(), DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_broadcast_to(p.allset, "compute_inertia", 0, &compute_inertia,
                              sizeof(int), DPU_XFER_DEFAULT));

  auto tic = std::chrono::steady_clock::now();
  //============RUNNING ONE LLOYD ITERATION ON THE DPU==============
  DPU_ASSERT(dpu_launch(p.allset, DPU_SYNCHRONOUS));
  //================================================================
  auto toc = std::chrono::steady_clock::now();
  p.time_seconds += std::chrono::duration<double>{toc - tic}.count();

  tic = std::chrono::steady_clock::now();
  /* copy back inertia (device to host) */
  DPU_FOREACH(p.allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &(inertia_psum[each_dpu])));
  }
  DPU_ASSERT(dpu_push_xfer(p.allset, DPU_XFER_FROM_DPU, "inertia", 0,
                           sizeof(inertia_psum[0]), DPU_XFER_DEFAULT));

  /* sum partial inertia */
  int64_t inertia =
      std::accumulate(inertia_psum.cbegin(), inertia_psum.cend(), 0LL);

  compute_inertia = 0;

  DPU_ASSERT(dpu_broadcast_to(p.allset, "compute_inertia", 0, &compute_inertia,
                              sizeof(int), DPU_XFER_DEFAULT));

  toc = std::chrono::steady_clock::now();

  p.pim_cpu_time = std::chrono::duration<double>{toc - tic}.count();

  return inertia;
}
