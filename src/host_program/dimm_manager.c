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
 * @brief Computes the greatest common divisor of two integers.
 *
 * @param i First integer.
 * @param j Second integer.
 * @return Their greatest common divisor.
 */
static int get_gcd(int i, int j) {
  while (j != 0) {
    int temp = i;
    i = j;
    j = temp % j;
  }
  return i;
}

/**
 * @brief Computes the lowest common multiple of two integers.
 *
 * @param i First integer.
 * @param j Second integer.
 * @return Their lowest common multiple.
 */
static int get_lcm(int i, int j) {
  return (int)((int64_t)i * j) / get_gcd(i, j);
}
