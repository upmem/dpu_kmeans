/**
 * @file kmeans.h
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for the KMeans project
 * @copyright 2021 UPMEM
 */

#ifndef _H_KMEANS
#define _H_KMEANS /**< header guard */

#ifndef _KMEANS_DPU_KERNEL_H_
#include <dpu.h>
#include <stdint.h>
#include <sys/time.h>

typedef struct dpu_set_t dpu_set;

#endif  // ifndef _KMEANS_DPU_KERNEL_H_

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif  // ifndef FLT_MAX

/** @name MEMsize
 * @brief DIMM memory sizes
 */
/**@{*/
#define MRAM_SIZE 67108864 /**< MRAM size per DPU in bytes */
#define WRAM_SIZE 65536    /**< WRAM size per DPU in bytes */
/**@}*/

/** @name Constraints
 * @brief Data size constraints
 */
/**@{*/
#define ASSUMED_NR_CLUSTERS 32 /**< Maximum number of clusters */
#define ASSUMED_NR_FEATURES 34 /**< Maximum number of features */
#define WRAM_FEATURES_SIZE \
  512 /**< max size of the WRAM array holding points features in bytes */
/**@}*/

// Whether or not to keep track of the memberships and do a final float
// reduction on the CPU #define FLT_REDUCE /**< Instruction to perform a float
// reduction on the CPU

// Performance tracking
// #define PERF_COUNTER 0 /* 0 for cycles, 1 for instructions, comment for no
// counter */

#ifdef PERF_COUNTER
#define HOST_COUNTERS 8
#define LOCAL_COUNTERS 13
/**
 * Enum used to index various performance counters
 */
enum perfcounter_names {
  TOTAL_CTR,
  MAIN_LOOP_CTR,
  CRITLOOP_ARITH_CTR,
  REDUCE_ARITH_CTR,
  REDUCE_LOOP_CTR,
  DISPATCH_CTR,
  MUTEX_CTR,
  INIT_CTR,
  ARITH_TIC,
  LOOP_TIC,
  MAIN_TIC,
  DISPATCH_TIC,
  MUTEX_TIC
};
#endif  // ifdef PERF_COUNTER

// Define the size of discretized features
// choose the value here, which is then propagated to the rest of the build,
// valid values are 8, 16 and 32.
#define FEATURE_TYPE 16

#if FEATURE_TYPE == 8
typedef int8_t int_feature;
#elif FEATURE_TYPE == 16
typedef int16_t int_feature;
#elif FEATURE_TYPE == 32
typedef int32_t int_feature;
#endif

#define MAX_FEATURE_DPU                                                   \
  (MRAM_SIZE / FEATURE_TYPE * 8 /                                         \
   2) /**< How many features we fit into one DPU's MRAM. Can be increased \
         further. */

#ifndef _KMEANS_DPU_KERNEL_H_
/**
 * @brief Struct holding various algorithm parameters.
 *
 */
typedef struct Params {
  uint64_t npoints;      /**< Number of points */
  uint64_t npadded;      /**< Number of points with padding */
  uint64_t npointperdpu; /**< Number of points per dpu */
  int nfeatures;         /**< Number of features */
  size_t nclusters;      /**< Number of clusters */
  int isOutput;          /**< Whether to print debug information */
  uint32_t ndpu;         /**< Number of allocated dpu */
  dpu_set allset;        /**< Struct of the allocated dpu set */
  double time_seconds;   /**< Perf counter */
  double cpu_pim_time;   /**< Time to populate the DPUs */
  double pim_cpu_time;   /**< Time to transfer inertia from the CPU */
} Params;

/* Function declarations */

/** @name dimmm_manager.c */
/**@{*/
void allocate_dpus(Params *p);
void free_dpus(Params *p);
void load_kernel(Params *p, const char *DPU_BINARY);
void build_jagged_array_int(uint64_t x_size, size_t y_size, int_feature *data,
                            int_feature ***features_out);
void broadcastNumberOfClusters(Params *p, size_t nclusters);
void populateDpu(Params *p, int_feature **feature);
void broadcastParameters(Params *p);
/**@}*/

/** @name lloyd_iter.c */
/**@{*/
void lloydIter(Params *p, int_feature *old_centers, int64_t *new_centers,
               int *new_centers_len, int *centers_pcount,
               int64_t *centers_psum);
uint64_t lloydIterWithInertia(Params *p, int_feature *old_centers,
                              uint64_t *inertia_psum);
/**@}*/
#endif  // ifndef _KMEANS_DPU_KERNEL_H_

#endif  // ifndef _H_KMEANS
