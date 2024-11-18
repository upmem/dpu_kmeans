/**
 * @file kmeans_dpu_kernel.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief DPU side of the KMeans algorithm.
 * @copyright 2021 UPMEM
 */

#ifndef _KMEANS_DPU_KERNEL_H_
#define _KMEANS_DPU_KERNEL_H_ /**< guard to prevent linking with CPU binaries \
                               */

#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <stdio.h>
#include <string.h>

#include "common.h"

#define MIN(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;      \
  })

/*================== VARIABLES ==========================*/
/*------------------ LOCAL ------------------------------*/
/** @name Globals
 * Global variables shared between tasklets
 */
/**@{*/
int itask_in_features;
int itask_in_points;
// unsigned int cluster_transfer_size;
uint8_t nclusters;
uint16_t ncluster_features;
/**@}*/

/*------------------ INPUT ------------------------------*/
/** @name Host
 * Variables for host application communication
 */
/**@{*/
__host struct task_parameters p_h;
__host int nclusters_host;
__host int npoints;
__host int compute_inertia = 0;
// __host unsigned int membership_size_in_bytes;
/**@}*/

/*------------------ OUTPUT -----------------------------*/
/**
 * @name Variable outputs
 * Variables for host output communication
 */
/**@{*/
__host uint64_t inertia;
/**@}*/

/*================== TABLES =============================*/
/*------------------ LOCAL ------------------------------*/
/* making all the tables fixed size for performance */

/** @name Lookup
 * Lookup tables to avoid repeated index computation
 */
/**@{*/
/** lookup table to quickly find the base index of a cluster */
uint16_t cluster_base_indices[ASSUMED_NR_CLUSTERS];
/** lookup table to quickly find the base index of a point */
uint16_t point_base_indices[WRAM_FEATURES_SIZE / sizeof(int_feature)];
/**@}*/

/** @name Tasklets
 * Tasklet tables
 */
/**@{*/
/** table to hold the inertia computed by each tasklet */
__dma_aligned uint64_t inertia_tasklets[NR_TASKLETS];
/**@}*/

/*------------------ INPUT ------------------------------*/
/** @name Input
 * Arrays receiving input data
 */
/**@{*/
/** centroids coordinates */
__host int_feature c_clusters[ASSUMED_NR_CLUSTERS * ASSUMED_NR_FEATURES];
/** array holding the data points (input) */
__mram_noinit int_feature t_features[MAX_FEATURE_DPU];
/**@}*/

/*------------------ OUTPUT ------------------------------*/
/** @name Output
 * Output arrays
 */
/**@{*/
// __mram_noinit int32_t c_clusters_mram[ASSUMED_NR_CLUSTERS *
// ASSUMED_NR_FEATURES]; (off because of MRAM transfer bug)
__dma_aligned int centers_count[ASSUMED_NR_CLUSTERS];
__mram_noinit int centers_count_mram[ASSUMED_NR_CLUSTERS];
//   int centers_count_tasklets[NR_TASKLETS][ASSUMED_NR_CLUSTERS];

__dma_aligned int64_t centers_sum[ASSUMED_NR_CLUSTERS * ASSUMED_NR_FEATURES];
__mram_noinit int64_t
    centers_sum_mram[ASSUMED_NR_CLUSTERS * ASSUMED_NR_FEATURES];
// __mram_noinit int64_t centers_sum_tasklets[NR_TASKLETS][ASSUMED_NR_CLUSTERS *
// ASSUMED_NR_FEATURES];
/**@}*/

/*================== SYNCHRONIZATION =====================*/
BARRIER_INIT(sync_barrier, NR_TASKLETS)
MUTEX_INIT(task_mutex);
MUTEX_INIT(write_mutex);
MUTEX_INIT(write_count_mutex);

/*================== PERFORMANCE TRACKING ================*/
#ifdef PERF_COUNTER
__host perfcounter_t host_counters[HOST_COUNTERS];
uint8_t active_tasklets;

MUTEX_INIT(perf_mutex);
#endif

/*================== FUNCTIONS ==========================*/
/**
 * @brief Assigns jobs to each tasklet.
 *
 * @param current_itask_in_points [out] Assigned task index (point index).
 * @param current_itask_in_features [out] Assgned task index (feature index).
 * @return true : There are jobs left to perform.
 * @return false : All jobs are over.
 */
#ifndef PERF_COUNTER
static inline bool taskDispatch(int *current_itask_in_points,
                                int *current_itask_in_features) {
#else
bool taskDispatch(int *current_itask_in_points, int *current_itask_in_features,
                  perfcounter_t *tasklet_counters) {
  tasklet_counters[DISPATCH_TIC] = perfcounter_get();
#endif
  mutex_lock(task_mutex);

  // load current task index
  *current_itask_in_points = itask_in_points;
  *current_itask_in_features = itask_in_features;

  // update the index
  itask_in_points += p_h.task_size_in_points;
  itask_in_features += p_h.task_size_in_features;

  mutex_unlock(task_mutex);

#ifdef PERF_COUNTER
  tasklet_counters[DISPATCH_CTR] +=
      perfcounter_get() - tasklet_counters[DISPATCH_TIC];
#endif

  return *current_itask_in_points < npoints;
}

/**
 * @brief Initializes all variables before a run.
 *
 * @param tasklet_id Id of the tasklet calling this function.
 */
static inline void initialize(uint8_t tasklet_id) {
  if (tasklet_id == 0) {
#ifdef PERF_COUNTER
    perfcounter_config((PERF_COUNTER) ? COUNT_INSTRUCTIONS : COUNT_CYCLES,
                       true);

    // initializing global perf counters
    memset(host_counters, 0, sizeof(host_counters));
    active_tasklets = 0;
#endif

    nclusters = (uint8_t)nclusters_host;
    ncluster_features = nclusters * p_h.nfeatures;

    // defining how much data we read/write from MRAM for each centroid
    // cluster_transfer_size = p_h.nfeatures * sizeof(**centers_sum_tasklets);
    // rounding cluster_transfer_size up to a multiple of 8
    // cluster_transfer_size = (cluster_transfer_size + 7) & -8;

    // counters initialization
    itask_in_features = 0;
    itask_in_points = 0;

    // // reading of c_clusters (currently off because of bugged transfert to
    // MRAM) int ncluster_features_even = (ncluster_features % 2 == 0) ?
    // ncluster_features : ncluster_features + 1; mram_read(c_clusters_mram,
    // c_clusters, ncluster_features * sizeof(int32_t));

    // reinitializing center counters and sums
    if (!compute_inertia) {
      memset(centers_count, 0, sizeof(*centers_count) * nclusters);
      memset(centers_sum, 0, sizeof(*centers_sum) * ncluster_features);
    }

    // reinitializing inertia table
    else {
      memset(inertia_tasklets, 0, sizeof(*inertia_tasklets) * NR_TASKLETS);
    }
  }

  barrier_wait(&sync_barrier);

  // pre-computing index lookup tables
  uint16_t cluster_base_index = tasklet_id * p_h.nfeatures;
  for (uint8_t icluster = tasklet_id; icluster < nclusters;
       icluster += NR_TASKLETS) {
    cluster_base_indices[icluster] = cluster_base_index;
    cluster_base_index += NR_TASKLETS * p_h.nfeatures;
  }

  uint16_t point_base_index = tasklet_id * p_h.nfeatures;
  for (uint8_t ipoint = tasklet_id; ipoint < p_h.task_size_in_points;
       ipoint += NR_TASKLETS) {
    point_base_indices[ipoint] = point_base_index;
    point_base_index += NR_TASKLETS * p_h.nfeatures;
  }

  // reinitializing center counters and sums per tasklet
  // memset(centers_count_tasklets[tasklet_id], 0,
  // sizeof(**centers_count_tasklets) * p_h.nclusters);
  // memset(centers_sum_tasklets[tasklet_id], 0, sizeof(**centers_sum_tasklets)
  // * ncluster_features);
}

/**
 * @brief Finds the cluster with the smallest distance to a point.
 * Used when a point fits in w_features.
 *
 * @param ipoint Point index relative to the current task.
 * @param min_dist [out] Distance to the closest cluster.
 * @param w_features Features vector for the current task.
 * @return uint8_t Index of the cluster with the smallest distance to the point.
 */
static inline uint8_t find_cluster_small_dim(
    uint8_t ipoint, uint64_t *min_dist,
    __dma_aligned const int_feature *w_features) {
  uint8_t index = UINT8_MAX;
  *min_dist = UINT64_MAX;
  uint16_t point_base_index = point_base_indices[ipoint];

#ifdef PERF_COUNTER
  tasklet_counters[ARITH_TIC] = perfcounter_get();
#endif
  /* find the cluster center id with min distance to pt */
  for (uint8_t icluster = 0; icluster < nclusters; icluster++) {
    uint64_t dist = 0; /* Euclidean distance squared */
    uint16_t cluster_base_index = cluster_base_indices[icluster];

#pragma clang loop unroll(enable)
    for (uint8_t idim = 0; idim < p_h.nfeatures; idim++) {
      volatile int_feature diff =
          (int_feature)(w_features[point_base_index + idim] -
                        c_clusters[cluster_base_index + idim]);
#if FEATURE_TYPE == 32
      dist += (uint64_t)((int64_t)diff * diff); /* sum of squares */
#else
      dist += (uint32_t)(diff * diff);            /* sum of squares */
#endif
    }
    /* see if distance is smaller than previous ones:
    if so, change minimum distance and save index of cluster center */
    if (dist < *min_dist) {
      *min_dist = dist;
      index = icluster;
    }
  }
#ifdef PERF_COUNTER
  tasklet_counters[CRITLOOP_ARITH_CTR] +=
      perfcounter_get() - tasklet_counters[ARITH_TIC];
#endif

  return index;
}

/**
 * @brief Sums the distances of each cluster to a partially loaded point.
 * Used when a single point does not fit in w_features.
 *
 * @param n_loaded_features Number of features loaded in w_features.
 * @param dists Array of distances to each cluster.
 * @param w_features Features vector for the current task.
 */
static inline void sum_clusters_large_dim(
    uint8_t n_loaded_features, uint64_t dists[static ASSUMED_NR_CLUSTERS],
    __dma_aligned const int_feature
        w_features[static WRAM_FEATURES_SIZE / sizeof(int_feature)]) {
  for (uint8_t icluster = 0; icluster < nclusters; icluster++) {
    uint16_t cluster_base_index = cluster_base_indices[icluster];

#pragma clang loop unroll(enable)
    for (uint8_t idim = 0; idim < n_loaded_features; idim++) {
      volatile int_feature diff =
          (int_feature)(w_features[idim] -
                        c_clusters[cluster_base_index + idim]);
#if FEATURE_TYPE == 32
      dist[icluster] += (uint64_t)((int64_t)diff * diff); /* sum of squares */
#else
      dists[icluster] += (uint32_t)(diff * diff); /* sum of squares */
#endif
    }
  }
}

/**
 * @brief Writes the result of each point to MRAM at the end of each point.
 *
 * @param tasklet_id Current tasklet id.
 * @param icluster Cluster the point is being assigned to.
 * @param point_global_index Index of the point in t_membership.
 * @param point_base_index Index of the point in the current task.
 * @param w_features Feature vector for the current task.
 */
#ifndef PERF_COUNTER
static inline void task_reduce_small_dim(uint8_t icluster,
                                         uint16_t point_base_index,
                                         const int_feature *w_features) {
#else
void task_reduce_small_dim(uint8_t icluster, uint16_t point_base_index,
                           int_feature *w_features,
                           perfcounter_t *tasklet_counters) {
  tasklet_counters[LOOP_TIC] = perfcounter_get();
#endif

  // centers_count_tasklets[tasklet_id][icluster]++;
  mutex_lock(write_count_mutex);
  centers_count[icluster]++;
  mutex_unlock(write_count_mutex);

  uint16_t cluster_base_index = cluster_base_indices[icluster];

#ifdef PERF_COUNTER
  tasklet_counters[ARITH_TIC] = perfcounter_get();
#endif
  mutex_lock(write_mutex);
#pragma clang loop unroll(enable)
  for (uint8_t idim = 0; idim < p_h.nfeatures; idim++) {
    // centers_sum_tasklets[tasklet_id][cluster_base_indices[icluster] + idim]
    // += w_features[point_base_index + idim];
    centers_sum[cluster_base_index + idim] +=
        w_features[point_base_index + idim];
  }
  mutex_unlock(write_mutex);
#ifdef PERF_COUNTER
  tasklet_counters[REDUCE_ARITH_CTR] +=
      perfcounter_get() - tasklet_counters[ARITH_TIC];
#endif

#ifdef PERF_COUNTER
  tasklet_counters[REDUCE_LOOP_CTR] +=
      perfcounter_get() - tasklet_counters[LOOP_TIC];
#endif
}

#ifndef PERF_COUNTER
static inline void task_reduce_large_dim(uint8_t icluster,
                                         int_feature *w_features) {
#else
void task_reduce_large_dim(uint8_t icluster, int_feature *w_features,
                           perfcounter_t *tasklet_counters) {
  tasklet_counters[LOOP_TIC] = perfcounter_get();
#endif

  mutex_lock(write_count_mutex);
  centers_count[icluster]++;
  mutex_unlock(write_count_mutex);

  uint16_t cluster_base_index = cluster_base_indices[icluster];

  for (int i = 0; i < p_h.task_size_in_features; i += WRAM_FEATURES_NR) {
    uint8_t n_loaded_features = (uint8_t)(p_h.task_size_in_features - i);
    uint32_t remaining_bytes = n_loaded_features * sizeof(int_feature);
    mram_read(&t_features[itask_in_features + i], w_features,
              MIN(WRAM_FEATURES_SIZE, remaining_bytes));

#ifdef PERF_COUNTER
    tasklet_counters[ARITH_TIC] = perfcounter_get();
#endif
    mutex_lock(write_mutex);
    for (uint8_t idim = 0; idim < n_loaded_features; idim++) {
      centers_sum[cluster_base_index + idim] += w_features[idim];
    }
    mutex_unlock(write_mutex);
#ifdef PERF_COUNTER
    tasklet_counters[REDUCE_ARITH_CTR] +=
        perfcounter_get() - tasklet_counters[ARITH_TIC];
#endif
  }

#ifdef PERF_COUNTER
  tasklet_counters[REDUCE_LOOP_CTR] +=
      perfcounter_get() - tasklet_counters[LOOP_TIC];
#endif
}

/**
 * @brief Finds the cluster with the smallest distance to a point.
 * Used when a point fits in w_features.
 *
 * @param tasklet_id Current tasklet id.
 * @param current_itask_in_points Current task index in points.
 * @param current_itask_in_features Current task index in features.
 * @param w_features Features vector for the current task.
 */
static inline void find_clusters_small_dim(
    uint8_t tasklet_id, int current_itask_in_points,
    int current_itask_in_features, __dma_aligned int_feature *w_features) {
  mram_read(&t_features[current_itask_in_features], w_features,
            p_h.task_size_in_bytes);

  uint8_t max_ipoint =
      (p_h.task_size_in_points < npoints - current_itask_in_points)
          ? p_h.task_size_in_points
          : (uint8_t)(npoints - current_itask_in_points);
  for (uint8_t ipoint = 0; ipoint < max_ipoint; ipoint++) {
    uint64_t min_dist;
    uint8_t index = find_cluster_small_dim(ipoint, &min_dist, w_features);
    uint16_t point_base_index = point_base_indices[ipoint];

#ifndef PERF_COUNTER
    if (!compute_inertia) {
      task_reduce_small_dim(index, point_base_index, w_features);
    } else {
      inertia_tasklets[tasklet_id] += min_dist;
    }
#else
    task_reduce_small_dim(index, point_base_index, w_features,
                          tasklet_counters);
#endif
  }

#ifdef PERF_COUNTER
  tasklet_counters[MAIN_LOOP_CTR] +=
      perfcounter_get() - tasklet_counters[MAIN_TIC];
#endif
}

/**
 * @brief Finds the cluster with the smallest distance to a point.
 * Used when a single point does not fit in w_features.
 *
 * @param tasklet_id Current tasklet id.
 * @param current_itask_in_points Current task index in points.
 * @param current_itask_in_features Current task index in features.
 * @param w_features Features vector for the current task.
 */
static inline void find_clusters_large_dim(
    uint8_t tasklet_id, int current_itask_in_features,
    __dma_aligned int_feature
        w_features[static WRAM_FEATURES_SIZE / sizeof(int_feature)]) {
  /* Euclidean distances squared for each cluster. */
  uint64_t dists[ASSUMED_NR_CLUSTERS] = {0};

  for (int i = 0; i < p_h.task_size_in_features; i += WRAM_FEATURES_NR) {
    uint8_t n_loaded_features = (uint8_t)(p_h.task_size_in_features - i);
    uint32_t remaining_bytes = n_loaded_features * sizeof(int_feature);
    mram_read(&t_features[current_itask_in_features + i], w_features,
              MIN(WRAM_FEATURES_SIZE, remaining_bytes));

    sum_clusters_large_dim(n_loaded_features, dists, w_features);
  }

  /* find the cluster center id with min distance to pt */
  uint8_t index = UINT8_MAX;
  uint64_t min_dist = UINT64_MAX;
  for (uint8_t icluster = 0; icluster < nclusters; icluster++) {
    if (dists[icluster] < min_dist) {
      min_dist = dists[icluster];
      index = icluster;
    }
  }

#ifndef PERF_COUNTER
  if (!compute_inertia) {
    task_reduce_large_dim(index, w_features);
  } else {
    inertia_tasklets[tasklet_id] += min_dist;
  }
#else
  task_reduce_small_dim(index, point_base_index, w_features, tasklet_counters);
#endif

#ifdef PERF_COUNTER
  tasklet_counters[MAIN_LOOP_CTR] +=
      perfcounter_get() - tasklet_counters[MAIN_TIC];
#endif
}

/**
 * @brief Final reduction: all tasklets work together to compute the partial
 * sums in WRAM.
 *
 * @param tasklet_id Current tasklet id.
 */
static inline void final_reduce(uint8_t tasklet_id) {
  // barrier_wait(&sync_barrier);

  // uint16_t cluster_base_index = tasklet_id * p_h.nfeatures;
  // #pragma must_iterate(1, ASSUMED_NR_CLUSTERS, 1)
  // for (uint8_t icluster = tasklet_id; icluster < p_h.nclusters; icluster +=
  // NR_TASKLETS)
  // {
  //     #pragma must_iterate(1, NR_TASKLETS, 1)
  //     for (uint8_t itasklet = 0; itasklet < NR_TASKLETS; itasklet++)
  //     {
  //         centers_count[icluster] +=
  //         centers_count_tasklets[itasklet][icluster];

  //         #pragma unroll(ASSUMED_NR_FEATURES)
  //         #pragma must_iterate(1, ASSUMED_NR_FEATURES, 1)
  //         for (uint8_t ifeature = 0; ifeature < p_h.nfeatures; ifeature++)
  //             centers_sum[cluster_base_index + ifeature] +=
  //             centers_sum_tasklets[itasklet][cluster_base_index +
  //             ifeature];
  //     }
  //     cluster_base_index += p_h.nfeatures * NR_TASKLETS;
  // }

  barrier_wait(&sync_barrier);

  if (tasklet_id == 0) {
    if (!compute_inertia) {
      // writing the partial sums and counts to MRAM
      uint16_t mram_transfer_size = nclusters * sizeof(*centers_count);
      // rounding up to multiple of 8
      mram_transfer_size = (mram_transfer_size + 7) & -8;
      mram_write(centers_count, centers_count_mram, mram_transfer_size);

      mram_transfer_size = ncluster_features * sizeof(*centers_sum);
      // rounding up to multiple of 8
      mram_transfer_size = (mram_transfer_size + 7) & -8;
      mram_write(centers_sum, centers_sum_mram, mram_transfer_size);
    } else {
      // summing inertia
      inertia = 0;
    }
    for (int i_tasklet = 0; i_tasklet < NR_TASKLETS; i_tasklet++) {
      inertia += inertia_tasklets[i_tasklet];
    }
  }
}

#ifdef PERF_COUNTER
/**
 * @brief Tallies all the performance counters from the tasklets.
 *
 * @param tasklet_id Current tasklet id.
 * @param tasklet_counters Current tasklet performance counters.
 */
void counters_tally(uint8_t tasklet_id, perfcounter_t *tasklet_counters) {
  mutex_lock(perf_mutex);

  // only counting tasklets that went through the main loop
  if (tasklet_counters[MAIN_LOOP_CTR] != 0) {
    active_tasklets++;
    for (int ictr = 0; ictr < HOST_COUNTERS; ictr++)
      host_counters[ictr] += tasklet_counters[ictr];
  }

  mutex_unlock(perf_mutex);

  barrier_wait(&sync_barrier);

  // averaging over active tasklets
  for (int ictr = tasklet_id; ictr < HOST_COUNTERS; ictr += NR_TASKLETS)
    if (active_tasklets) host_counters[ictr] /= active_tasklets;
}
#endif

/*================== MAIN FUNCTION ======================*/
/**
 * @brief Main function DPU side.
 *
 * @return 0 on success
 */
int main() {
  uint8_t tasklet_id = (uint8_t)me();

  int current_itask_in_points;
  int current_itask_in_features;

  /* dma transfers can go up to 2048 bytes
   * but we also need to not explode the stack */
  __dma_aligned int_feature
      w_features[WRAM_FEATURES_SIZE / sizeof(int_feature)];

#ifdef PERF_COUNTER
  perfcounter_t tasklet_counters[LOCAL_COUNTERS] = {0};
#endif

  initialize(tasklet_id);

  barrier_wait(&sync_barrier);

#ifndef PERF_COUNTER
  while (taskDispatch(&current_itask_in_points, &current_itask_in_features)) {
#else
  tasklet_counters[INIT_CTR] = perfcounter_get();

  while (taskDispatch(&current_itask_in_points, &current_itask_in_features,
                      tasklet_counters)) {
    tasklet_counters[MAIN_TIC] = perfcounter_get();
#endif

    if (p_h.task_size_in_bytes <= WRAM_FEATURES_SIZE) {
      find_clusters_small_dim(tasklet_id, current_itask_in_points,
                              current_itask_in_features, w_features);
    } else {
      find_clusters_large_dim(tasklet_id, current_itask_in_features,
                              w_features);
    }
  }

#ifdef PERF_COUNTER
  tasklet_counters[LOOP_TIC] = perfcounter_get();
#endif
  final_reduce(tasklet_id);
#ifdef PERF_COUNTER
  tasklet_counters[REDUCE_LOOP_CTR] +=
      perfcounter_get() - tasklet_counters[LOOP_TIC];
#endif

#ifdef PERF_COUNTER
  tasklet_counters[TOTAL_CTR] = perfcounter_get();

  counters_tally(tasklet_id, tasklet_counters);
#endif

  // DEBUG
  // barrier_wait(&sync_barrier);
  // if(tasklet_id==0) {
  //   printf("nreal_points: %d\n", npoints);

  //   // printf("maxes: ");
  //   // for(int ifeature = 0; ifeature < p_h.nfeatures; ifeature++){
  //   //     int64_t max_mean = 0;
  //   //     for(int icluster = 0; icluster < p_h.nclusters; icluster++){
  //   //         if(centers_count[icluster] > 0
  //   &&centers_sum[cluster_base_indices[icluster]+ifeature]
  //   /centers_count[icluster] > max_mean)
  //   //             max_mean =
  //   centers_sum[cluster_base_indices[icluster]+ifeature]/centers_count[icluster];
  //   //     }
  //   //     printf("%lld ", max_mean);
  //   // }
  //   // printf("\n");

  //   // for(int ipoint = 0; ipoint<npoints; ipoint++)
  //   // {
  //   //     printf("%d ", t_membership[ipoint]);
  //   // }
  //   // printf("\n");

  //   for(int ifeature = 0; ifeature< p_h.nfeatures; ifeature++)
  //   {
  //     printf("count cluster %d : %d\n", ifeature, centers_count[ifeature]);
  //   }

  //   // printf("counter for reduction = %ld\n",
  //   // host_counters[REDUCE_LOOP_CTR]);
  // }

  return 0;
}
#endif  // ifndef _KMEANS_DPU_KERNEL_H_
