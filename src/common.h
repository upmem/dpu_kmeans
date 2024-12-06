/**
 * @file common.h
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Common definitions for CPU and DPU code.
 * @copyright 2024 UPMEM
 */

#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

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
#define ASSUMED_NR_CLUSTERS 32  /**< Maximum number of clusters */
#define ASSUMED_NR_FEATURES 128 /**< Maximum number of features */
#define WRAM_FEATURES_SIZE \
  512 /**< max size of the WRAM array holding points features in bytes */
#define DMA_ALIGN 8U                /**< DMA alignment */
#define MAX_MRAM_TRANSFER_SIZE 2048 /**< Maximum size of a MRAM transfer */
/**@}*/

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

struct __attribute__((aligned(8))) task_parameters {
  uint8_t nfeatures;
  uint8_t task_size_in_points;
  uint16_t task_size_in_features;
  uint16_t task_size_in_bytes;
};
