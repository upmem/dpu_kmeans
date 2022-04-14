/**
 * @file lloy_iter.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Performs one iteration of the Lloyd K-Means algorithm.
 */

#include <dpu.h>
#include <stddef.h>

#include "../kmeans.h"

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
 * @brief Returns the seconds elapsed between two timeval structures.
 *
 * @param tic [in] First timeval.
 * @param toc [in] Second timeval.
 * @return double Elapsed time in seconds.
 */
static double time_seconds(struct timeval tic, struct timeval toc) {
  struct timeval timing;
  timing.tv_sec = toc.tv_sec - tic.tv_sec;
  timing.tv_usec = toc.tv_usec - tic.tv_usec;
  double time = ((double)(timing.tv_sec * 1000000 + timing.tv_usec)) / 1000000;

  return time;
}

/**
 * @brief Performs one iteration of the Lloyd algorithm on DPUs and gets the
 * results.
 */
void lloydIter(
    Params *p, /**< Algorithm parameters. */
    int_feature
        *old_centers, /**< [in] Discretized current centroids coordinates. */
    int64_t *new_centers, /**< [out] Discretized updated centroids coordinates
                             (before division by cluster count). */
    int *new_centers_len, /**< [out] Number of elements in each cluster. */
    int *centers_pcount,  /**< Buffer to read cluster counts per DPU. */
    int64_t *centers_psum /**< Buffer to read coordinates sum per DPU. */
) {
  struct dpu_set_t dpu;    /* Iteration variable for the DPUs. */
  uint32_t each_dpu;       /* Iteration variable for the DPUs. */
  struct timeval toc, tic; /* Perf counters */

#ifdef PERF_COUNTER
  uint64_t counters_mean[HOST_COUNTERS] = {0};
#endif

  DPU_ASSERT(dpu_broadcast_to(p->allset, "c_clusters", 0, old_centers,
                              p->nclusters * p->nfeatures * sizeof(int_feature),
                              DPU_XFER_DEFAULT));

  gettimeofday(&tic, NULL);
  //============RUNNING ONE LLOYD ITERATION ON THE DPU==============
  DPU_ASSERT(dpu_launch(p->allset, DPU_SYNCHRONOUS));
  //================================================================
  gettimeofday(&toc, NULL);
  p->time_seconds += time_seconds(tic, toc);

  /* DEBUG : read logs */
  // DPU_FOREACH(p->allset, dpu, each_dpu) {
  //     if (each_dpu >= 0)
  //         DPU_ASSERT(dpu_log_read(dpu, stdout));
  // }
  // exit(0);

  /* Performance tracking */
#ifdef PERF_COUNTER
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &counters[each_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "host_counters", 0,
                           sizeof(uint64_t[HOST_COUNTERS]), DPU_XFER_DEFAULT));

  for (int icounter = 0; icounter < HOST_COUNTERS; icounter++) {
    int nonzero_dpus = 0;
    for (int idpu = 0; idpu < p->ndpu; idpu++)
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
  size_t count_in_8bytes = 8 / sizeof(*centers_pcount);
  size_t nclusters_aligned =
      ((p->nclusters + count_in_8bytes - 1) / count_in_8bytes) *
      count_in_8bytes;
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    DPU_ASSERT(
        dpu_prepare_xfer(dpu, &(centers_pcount[each_dpu * nclusters_aligned])));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "centers_count_mram",
                           0, sizeof(*centers_pcount) * nclusters_aligned,
                           DPU_XFER_DEFAULT));

  /* DEBUG : print outputed centroids counts per DPU */
  // for (int dpu_id = 0; dpu_id < p->ndpu; dpu_id++) {
  //   for (int cluster_id = 0; cluster_id < p->nclusters; cluster_id++) {
  //     printf("%d ", centers_pcount[dpu_id * p->nclusters + cluster_id]);
  //   }
  //   printf("\n");
  // }

  /* copy back centroids partial averages (device to host) */
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(
        dpu,
        &centers_psum[offset(0, 0, each_dpu, p->nfeatures, p->nclusters)]));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "centers_sum_mram", 0,
                           p->nfeatures * p->nclusters * sizeof(int64_t),
                           DPU_XFER_DEFAULT));

  memset(new_centers, 0, p->nclusters * p->nfeatures * sizeof(*new_centers));
  memset(new_centers_len, 0, p->nclusters * sizeof(*new_centers_len));

  for (int dpu_id = 0; dpu_id < p->ndpu; dpu_id++) {
    for (int cluster_id = 0; cluster_id < p->nclusters; cluster_id++) {
      /* sum membership counts */
      new_centers_len[cluster_id] +=
          centers_pcount[dpu_id * nclusters_aligned + cluster_id];
      /* compute the new centroids sum */
      for (int feature_id = 0; feature_id < p->nfeatures; feature_id++)
        new_centers[cluster_id * p->nfeatures + feature_id] +=
            centers_psum[offset(feature_id, cluster_id, dpu_id, p->nfeatures,
                                p->nclusters)];
    }
  }

  /* average the new centers */
  /* this has been moved to the python code */
  // for (int cluster_id = 0; cluster_id < p->nclusters; cluster_id++) {
  //   if (new_centers_len[cluster_id])
  //     for (int feature_id = 0; feature_id < p->nfeatures; feature_id++) {
  //       new_centers[cluster_id * p->nfeatures + feature_id] /=
  //           new_centers_len[cluster_id];
  //     }
  // }

  /* DEBUG: print new clusters */
  // printf("new clusters :\n");
  // for (int cluster_id = 0; cluster_id < p->nclusters; cluster_id++) {
  //   for (int feature_id = 0; feature_id < p->nfeatures; feature_id++) {
  //     printf(new_centers[cluster_id * p->nclusters + feature_id])
  //   }
  // }
}

/**
 * @brief Performs one E step of the Lloyd algorithm on DPUs and gets the
 * inertia only.
 */
uint64_t lloydIterWithInertia(
    Params *p, /**< Algorithm parameters. */
    int_feature
        *old_centers, /**< [in] Discretized current centroids coordinates. */
    uint64_t *inertia_psum /**< Buffer to read inertia per DPU. */
) {
  struct dpu_set_t dpu;    /* Iteration variable for the DPUs. */
  uint32_t each_dpu;       /* Iteration variable for the DPUs. */
  struct timeval toc, tic; /* Perf counters */

  int compute_inertia = 1;

  DPU_ASSERT(dpu_broadcast_to(p->allset, "c_clusters", 0, old_centers,
                              p->nclusters * p->nfeatures * sizeof(int_feature),
                              DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_broadcast_to(p->allset, "compute_inertia", 0, &compute_inertia,
                              sizeof(int), DPU_XFER_DEFAULT));

  gettimeofday(&tic, NULL);
  //============RUNNING ONE LLOYD ITERATION ON THE DPU==============
  DPU_ASSERT(dpu_launch(p->allset, DPU_SYNCHRONOUS));
  //================================================================
  gettimeofday(&toc, NULL);
  p->time_seconds += time_seconds(tic, toc);

  gettimeofday(&tic, NULL);
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

  gettimeofday(&toc, NULL);

  p->pim_cpu_time = time_seconds(tic, toc);

  return inertia;
}
