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
#include <dpu.h>

typedef struct dpu_set_t dpu_set;

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

#ifndef _KMEANS_DPU_KERNEL_H_
// Parameters holding struct
typedef struct Params{
    uint64_t npoints;
    uint64_t npadded;
    uint64_t npointperdpu;
    int nfeatures;
    float scale_factor;
    float threshold;
    float *mean;
    int max_nclusters;
    int min_nclusters;
    int isRMSE;
    int isOutput;
    int nloops;
    int max_iter;
    uint32_t ndpu;
    dpu_set allset;
    int from_file;
} Params;

// Function declarations
/** @name rmse.c */
/**@{*/
float rms_err(Params *p, float **feature, float **cluster_centres, int nclusters);
/**@}*/

/** @name kmeans.c */
/**@{*/
void allocate(Params *p);
double time_seconds(struct timeval tic, struct timeval toc);
void read_bin_input(
    Params *p,
    const char *filename,
    float ***features_out);
void read_txt_input(
    Params *p,
    const char *filename,
    float ***features_out);
void save_dat_file(Params *p, const char *filename_in, float **features);
void format_array_input(
    Params *p,
    float *data,
    float ***features_out);
void preprocessing(
    Params *p,
    float **features,
    int_feature ***features_int_out,
    int verbose);
void postprocessing(Params *p, float **features);
float *kmeans_c(
    Params *p,
    float **features_float,
    int_feature **features_int,
    int *log_iterations,
    double *log_time,
    int *best_nclusters);
/**@}*/

/** @name cluster.c */
/**@{*/
void load_kernel(Params *p, const char *DPU_BINARY);
void free_dpus(Params *p);
int cluster(
    Params *p,
    float **features_float,
    int_feature **features_int,
    int *best_nclusters,
    float ***cluster_centres,
    float *min_rmse,
    int *log_iterations,
    double *log_time);
/**@}*/

/** @name kmeans_clustering.c */
/**@{*/
void allocateClusters(Params *p, unsigned int nclusters);
void deallocateClusters();
float **kmeans_clustering(
    Params *p,
    int_feature **features_int,
    float **features_float,
    unsigned int nclusters,
    int *loop,
    int i_init);
/**@}*/

/** @name kmeans_dpu.c */
/**@{*/
void allocateMemory(Params *p);
void deallocateMemory();
void populateDpu(
    Params *p,
    int_feature **feature);
void kmeansDpu(
    Params *p,
    int nclusters,
    int64_t new_centers_len[ASSUMED_NR_CLUSTERS],
    int64_t new_centers[ASSUMED_NR_CLUSTERS][ASSUMED_NR_FEATURES]);
/**@}*/
#endif // ifndef _KMEANS_DPU_KERNEL_H_

#endif // ifndef _H_KMEANS
