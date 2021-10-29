/**
 * @file kmeans.h
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for the KMeans project
 *
 */

#ifndef _H_KMEANS
#define _H_KMEANS /**< header guard */

#ifndef _KMEANS_DPU_KERNEL_H_
#include <stdint.h>
#include <sys/time.h>
#endif // ifndef _KMEANS_DPU_KERNEL_H_

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif // ifndef FLT_MAX

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
#define MAX_FEATURE_DPU 5000000 /**< How many features we fit into one DPU's MRAM. Can be increased further. */
#define ASSUMED_NR_CLUSTERS 32  /**< Maximum number of clusters */
#define ASSUMED_NR_FEATURES 34  /**< Maximum number of features */
#define WRAM_FEATURES_SIZE 512  /**< max size of the WRAM array holding points features in bytes */
/**@}*/

// Whether or not to keep track of the memberships and do a final float reduction on the CPU
// #define FLT_REDUCE /**< Instruction to perform a float reduction on the CPU

// Performance tracking
// #define PERF_COUNTER 0 /* 0 for cycles, 1 for instructions, comment for no counter */

#ifdef PERF_COUNTER
#define HOST_COUNTERS 8
#define LOCAL_COUNTERS 13
/**
 * Enum used to index various performance counters
 */
enum perfcounter_names
{
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
#endif // ifdef PERF_COUNTER

// Define the size of discretized features (choose one):

// typedef int8_t int_feature;
////////// OR
typedef int16_t int_feature;
////////// OR
// typedef int32_t int_feature;
// #define FEATURETYPE_32

// Function declarations
#ifndef _KMEANS_DPU_KERNEL_H_
/** @name rmse.c */
/**@{*/
float euclid_dist_2(float *, float *, int);
int find_nearest_point(float *, int, float **, int);
float rms_err(float **, int, uint64_t, float **, int);
/**@}*/

/** @name kmeans.c */
/**@{*/
double time_seconds(struct timeval, struct timeval);
void strip_ext(char *);
void usage(char *);
float preprocessing(float **, int, uint64_t, uint64_t, float **, int_feature ***, float *);
void read_binary_input(char *, uint64_t *, uint64_t *, int *, uint32_t, float ***);
void read_text_input(char *, uint64_t *, uint64_t *, int *, uint32_t, float ***);
int main(int, char **);
/**@}*/

/** @name cluster.c */
/**@{*/
int cluster(uint64_t, uint64_t, int, uint32_t, float **, int_feature **, int, int, float, int *, float ***, float *, int, int, char *, const char *);
int get_lcm(int, int);
/**@}*/

/** @name kmeans_clustering.c */
/**@{*/
float **kmeans_clustering(int_feature **, float **, int, uint64_t, uint64_t, unsigned int, int, float, uint8_t *, int *, int);
void final_reduction(float **, int, uint64_t, uint64_t, unsigned int, int, uint8_t *, float **);
/**@}*/

/** @name kmeans_dpu.c */
/**@{*/
void allocateMemory(uint64_t, int);
void deallocateMemory();
void populateDpu(int_feature **, int, uint64_t, uint64_t, int);
void kmeansDpu(int, uint64_t, uint64_t, int, int, int64_t[ASSUMED_NR_CLUSTERS], int64_t[ASSUMED_NR_CLUSTERS][ASSUMED_NR_FEATURES]);
/**@}*/
#endif // ifndef _KMEANS_DPU_KERNEL_H_

#endif // ifndef _H_KMEANS
