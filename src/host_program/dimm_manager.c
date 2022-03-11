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
void allocate_dpus(Params *p) {
  if (!p->ndpu)
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &p->allset));
  else
    DPU_ASSERT(dpu_alloc(p->ndpu, NULL, &p->allset));
  DPU_ASSERT(dpu_get_nr_dpus(p->allset, &p->ndpu));
}

/**
 * @brief Frees the DPUs.
 *
 * @param p Algorithm parameters.
 */
void free_dpus(Params *p) { DPU_ASSERT(dpu_free(p->allset)); }

/**
 * @brief Loads a binary in the DPUs.
 *
 * @param p Algorithm parameters.
 * @param DPU_BINARY path to the binary
 */
void load_kernel(Params *p, const char *DPU_BINARY) {
  DPU_ASSERT(dpu_load(p->allset, DPU_BINARY, NULL));
}

/**
 * @brief Formats a flat integer array into a bidimensional representation.
 */
void build_jagged_array_int(
    uint64_t x_size,             /**< [in] Size of the first dimension. */
    size_t y_size,               /**< [in] Size of the second dimension. */
    int_feature *data,           /**< [in] The data as a flat table */
    int_feature ***features_out) /**< [out] The data as two dimensional table */
{
  int_feature **features = (int_feature **)malloc(x_size * sizeof(*features));
  features[0] = data;
  for (int ipoint = 1; ipoint < x_size; ipoint++)
    features[ipoint] = features[ipoint - 1] + y_size;

  *features_out = features;
}

/**
 * @brief Broadcast current number of clusters to the DPUs
 *
 * @param p Algorithm parameters.
 * @param nclusters Number of clusters.
 */
void broadcastNumberOfClusters(Params *p, size_t nclusters) {
  /* inform DPUs of the current number of clusters */
  unsigned int nclusters_short = nclusters;
  DPU_ASSERT(dpu_broadcast_to(p->allset, "nclusters_host", 0, &nclusters_short,
                              sizeof(nclusters_short), DPU_XFER_DEFAULT));
}

/**
 * @brief Fills the DPUs with their assigned points.
 */
void populateDpu(Params *p,             /**< Algorithm parameters */
                 int_feature **feature) /**< array: [npoints][nfeatures] */
{
  /* Iteration variables for the DPUs. */
  struct dpu_set_t dpu;
  uint32_t each_dpu;

  int *nreal_points; /* number of real data points on each dpu */
  int64_t remaining_points = p->npoints; /* number of yet unassigned points */

  DPU_FOREACH(p->allset, dpu, each_dpu) {
    int next;
    next = each_dpu * p->npointperdpu;
    DPU_ASSERT(dpu_prepare_xfer(dpu, feature[next]));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "t_features", 0,
                           p->npointperdpu * p->nfeatures * sizeof(int_feature),
                           DPU_XFER_DEFAULT));

  // telling each DPU how many real points it has to process
  nreal_points = (int *)malloc(p->ndpu * sizeof(*nreal_points));
  for (int idpu = 0; idpu < p->ndpu; idpu++) {
    nreal_points[idpu] = (remaining_points <= 0) ? 0
                         : (remaining_points > p->npointperdpu)
                             ? p->npointperdpu
                             : remaining_points;
    remaining_points -= p->npointperdpu;
  }

  /* DEBUG : print the number of non-padding points assigned to each DPU */
  // printf("nreal_points :\n");
  // for(int idpu = 0; idpu < ndpu; idpu++)
  // {
  //     printf("%d ", nreal_points[idpu]);
  // }
  // printf("\n");

  DPU_FOREACH(p->allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &nreal_points[each_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "npoints", 0,
                           sizeof(int), DPU_XFER_DEFAULT));
  free(nreal_points);
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
static unsigned int get_task_size(Params *p) {
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
void broadcastParameters(Params *p) {
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
