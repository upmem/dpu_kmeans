/**
 * @file dimm_manager.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Nanny functions for the DPUs.
 *
 */

#include "../kmeans.h"

/**
 * @brief Allocates all DPUs
 *
 * @param p Algorithm parameters.
 */
void allocate_dpus(kmeans_params *p) {
  if (!p->ndpu) {
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &p->allset));
  } else {
    DPU_ASSERT(dpu_alloc(p->ndpu, NULL, &p->allset));
  }
  DPU_ASSERT(dpu_get_nr_dpus(p->allset, &p->ndpu));
}

/**
 * @brief Frees the DPUs.
 *
 * @param p Algorithm parameters.
 */
void free_dpus(kmeans_params *p) { DPU_ASSERT(dpu_free(p->allset)); }

/**
 * @brief Loads a binary in the DPUs.
 *
 * @param p Algorithm parameters.
 * @param DPU_BINARY path to the binary
 */
void load_kernel(kmeans_params *p, const char *binary_path) {
  DPU_ASSERT(dpu_load(p->allset, binary_path, NULL));
}

/**
 * @brief Broadcast current number of clusters to the DPUs
 *
 * @param p Algorithm parameters.
 * @param nclusters Number of clusters.
 */
void broadcastNumberOfClusters(kmeans_params *p, size_t nclusters) {
  /* inform DPUs of the current number of clusters */
  unsigned int nclusters_short = nclusters;
  DPU_ASSERT(dpu_broadcast_to(p->allset, "nclusters_host", 0, &nclusters_short,
                              sizeof(nclusters_short), DPU_XFER_DEFAULT));
}

/**
 * @brief Computes the lowest common multiple of two integers.
 *
 * @param n1 First integer.
 * @param n2 Second integer.
 * @return Their lowest common multiple.
 */
static int get_lcm(int n1, int n2) {
  static int max = 1;
  if (max % n1 == 0 && max % n2 == 0) {
    return max;
  } else {
    max++;
    get_lcm(n1, n2);
    return max;
  }
}

/**
 * @brief Computes the appropriate task size for DPU tasklets.
 *
 * @param p Algorithm parameters.
 * @return The task size in bytes.
 */
static unsigned int get_task_size(kmeans_params *p) {
  unsigned int task_size_in_points;
  unsigned int task_size_in_bytes;
  unsigned int task_size_in_features;

  /* how many points we can fit in w_features */
  unsigned int max_task_size =
      (WRAM_FEATURES_SIZE / sizeof(int_feature)) / p->nfeatures;

  /* number of tasks as the smallest multiple of NR_TASKLETS higher than
   * npointperdu / max_task_size */
  unsigned int ntasks = (p->npointperdpu + max_task_size - 1) / max_task_size;
  ntasks = ((ntasks + NR_TASKLETS - 1) / NR_TASKLETS) * NR_TASKLETS;

  /* task size has to be at least 1 */
  task_size_in_points =
      (((p->npointperdpu + ntasks - 1) / ntasks) < max_task_size)
          ? ((p->npointperdpu + ntasks - 1) / ntasks)
          : max_task_size;
  if (task_size_in_points == 0) task_size_in_points = 1;

  task_size_in_features = task_size_in_points * p->nfeatures;
  task_size_in_bytes = task_size_in_features * sizeof(int_feature);

  /* task size in bytes must be a multiple of 8 for DMA alignment and also a
   * multiple of number of features x byte size of integers */
  int lcm = get_lcm(sizeof(int_feature) * p->nfeatures, 8);
  task_size_in_bytes = task_size_in_bytes / lcm * lcm;
  if (task_size_in_bytes > WRAM_FEATURES_SIZE) {
    printf("error: tasks will not fit in WRAM");
    exit(EXIT_FAILURE);
  }
  /* minimal size */
  if (task_size_in_bytes < lcm) task_size_in_bytes = lcm;

  return task_size_in_bytes;
}

/**
 * @brief Broadcasts iteration parameters to the DPUs.
 *
 * @param p Algorithm parameters.
 */
void broadcastParameters(kmeans_params *p) {
  /* parameters to calculate once here and send to the DPUs. */
  unsigned int task_size_in_points;
  unsigned int task_size_in_bytes;
  unsigned int task_size_in_features;

  /* compute the iteration variables for the DPUs */

  task_size_in_bytes = get_task_size(p);

  /* realign task size in features and points */
  task_size_in_features = task_size_in_bytes / sizeof(int_feature);
  task_size_in_points = task_size_in_features / p->nfeatures;

  /* send computation parameters to the DPUs */
  DPU_ASSERT(dpu_broadcast_to(p->allset, "nfeatures_host", 0, &p->nfeatures,
                              sizeof(p->nfeatures), DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_points_host", 0,
                              &task_size_in_points, sizeof(task_size_in_points),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_bytes_host", 0,
                              &task_size_in_bytes, sizeof(task_size_in_bytes),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_features_host", 0,
                              &task_size_in_features,
                              sizeof(task_size_in_features), DPU_XFER_DEFAULT));

  if (p->isOutput) {
    printf("points per DPU : %lu\n", p->npointperdpu);
    printf("task size in points : %u\n", task_size_in_points);
    printf("task size in bytes : %u\n", task_size_in_bytes);
    printf("tasks per DPU: %lu\n", p->npointperdpu / task_size_in_points);
  }
}
