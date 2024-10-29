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

#ifdef PERF_COUNTER
#include <array>
#endif

extern "C" {
#include "../kmeans.h"
}

/**
 * @brief Utility function for three dimensional arrays.
 *
 * @param feature current feature
 * @param cluster current cleaster
 * @param dpu current dpu
 * @param nfeatures number of features
 * @param nclusters number of clusters
 * @return array index
 */
static int offset(int feature, int cluster, int dpu, int nfeatures,
                  int nclusters) {
  return (dpu * nclusters * nfeatures) + (cluster * nfeatures) + feature;
}

/**
 * @brief Performs one iteration of the Lloyd algorithm on DPUs and gets the
 * results.
 *
 * @param p Algorithm parameters.
 * @param old_centers [in] Discretized current centroids coordinates.
 * @param new_centers [out] Discretized updated centroids coordinates (before
 * division by cluster count).
 * @param new_centers_len [out] Number of elements in each cluster.
 * @param centers_pcount Buffer to read cluster counts per DPU.
 * @param centers_psum Buffer to read coordinates sum per DPU.
 */
void lloydIter(kmeans_params &p, const py::array_t<int_feature> &old_centers,
               py::array_t<int64_t> &new_centers,
               py::array_t<int> &new_centers_len,
               std::vector<int> &centers_pcount,
               std::vector<int64_t> &centers_psum) {
  dpu_set_t dpu{};       /* Iteration variable for the DPUs. */
  uint32_t each_dpu = 0; /* Iteration variable for the DPUs. */

#ifdef PERF_COUNTER
  std::array<uint64_t, HOST_COUNTERS> counters_mean = {};
  std::vector<std::array<uint64_t, HOST_COUNTERS>> counters(p.ndpu);
#endif

  DPU_ASSERT(dpu_broadcast_to(p.allset, "c_clusters", 0, old_centers.data(0, 0),
                              p.nclusters * p.nfeatures * sizeof(int_feature),
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
  size_t count_in_8bytes = 8 / sizeof(centers_pcount.back());
  size_t nclusters_aligned =
      ((p.nclusters + count_in_8bytes - 1) / count_in_8bytes) * count_in_8bytes;
  DPU_FOREACH(p.allset, dpu, each_dpu) {
    DPU_ASSERT(
        dpu_prepare_xfer(dpu, &(centers_pcount[each_dpu * nclusters_aligned])));
  }
  DPU_ASSERT(dpu_push_xfer(p.allset, DPU_XFER_FROM_DPU, "centers_count_mram", 0,
                           sizeof(centers_pcount.back()) * nclusters_aligned,
                           DPU_XFER_DEFAULT));

  /* copy back centroids partial averages (device to host) */
  DPU_FOREACH(p.allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(
        dpu, &centers_psum[offset(0, 0, each_dpu, p.nfeatures, p.nclusters)]));
  }
  DPU_ASSERT(dpu_push_xfer(
      p.allset, DPU_XFER_FROM_DPU, "centers_sum_mram", 0,
      static_cast<long>(p.nfeatures) * p.nclusters * sizeof(int64_t),
      DPU_XFER_DEFAULT));

  new_centers[py::make_tuple(py::ellipsis())] = 0LL;
  new_centers_len[py::make_tuple(py::ellipsis())] = 0;

  for (int dpu_id = 0; dpu_id < p.ndpu; dpu_id++) {
    for (int cluster_id = 0; cluster_id < p.nclusters; cluster_id++) {
      /* sum membership counts */
      new_centers_len.mutable_at(cluster_id) +=
          centers_pcount[dpu_id * nclusters_aligned + cluster_id];
      /* compute the new centroids sum */
      for (int feature_id = 0; feature_id < p.nfeatures; feature_id++) {
        new_centers.mutable_at(cluster_id, feature_id) += centers_psum[offset(
            feature_id, cluster_id, dpu_id, p.nfeatures, p.nclusters)];
      }
    }
  }

  /* averaging the new centers
   * has been moved to the python code */
}

/**
 * @brief Performs one E step of the Lloyd algorithm on DPUs and gets the
 * inertia only.
 */
uint64_t lloydIterWithInertia(
    kmeans_params *p, /**< Algorithm parameters. */
    int_feature
        *old_centers, /**< [in] Discretized current centroids coordinates. */
    int64_t *inertia_psum /**< Buffer to read inertia per DPU. */
) {
  struct dpu_set_t dpu; /* Iteration variable for the DPUs. */
  uint32_t each_dpu;    /* Iteration variable for the DPUs. */

  int compute_inertia = 1;

  DPU_ASSERT(dpu_broadcast_to(p->allset, "c_clusters", 0, old_centers,
                              p->nclusters * p->nfeatures * sizeof(int_feature),
                              DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_broadcast_to(p->allset, "compute_inertia", 0, &compute_inertia,
                              sizeof(int), DPU_XFER_DEFAULT));

  auto tic = std::chrono::steady_clock::now();
  //============RUNNING ONE LLOYD ITERATION ON THE DPU==============
  DPU_ASSERT(dpu_launch(p->allset, DPU_SYNCHRONOUS));
  //================================================================
  auto toc = std::chrono::steady_clock::now();
  p->time_seconds += std::chrono::duration<double>{toc - tic}.count();

  tic = std::chrono::steady_clock::now();
  /* copy back inertia (device to host) */
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &(inertia_psum[each_dpu])));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "inertia", 0,
                           sizeof(*inertia_psum), DPU_XFER_DEFAULT));

  /* sum partial inertia */
  uint64_t inertia = 0;
  for (int dpu_id = 0; dpu_id < p->ndpu; dpu_id++) {
    inertia += inertia_psum[dpu_id];
  }

  compute_inertia = 0;

  DPU_ASSERT(dpu_broadcast_to(p->allset, "compute_inertia", 0, &compute_inertia,
                              sizeof(int), DPU_XFER_DEFAULT));

  toc = std::chrono::steady_clock::now();

  p->pim_cpu_time = std::chrono::duration<double>{toc - tic}.count();

  return inertia;
}
