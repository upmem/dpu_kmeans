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

// Function declarations
#ifndef _KMEANS_DPU_KERNEL_H_
/** @name rmse.c */
/**@{*/
float rms_err(float **, int, uint64_t, float **, int);
/**@}*/

/** @name kmeans.c */
/**@{*/
double time_seconds(struct timeval tic, struct timeval toc);
void read_bin_input(
    const char *filename,
    uint64_t *npoints_out,
    uint64_t *npadded_out,
    int *nfeatures_out,
    uint32_t ndpu,
    float ***features_out);
void read_txt_input(
    const char *filename,
    uint64_t *npoints_out,
    uint64_t *npadded_out,
    int *nfeatures_out,
    uint32_t ndpu,
    float ***features_out);
void save_dat_file(const char *filename_in, uint64_t npoints, int nfeatures, float **features);
void format_array_input(
    uint64_t npoints,
    uint64_t *npadded_out,
    int nfeatures,
    uint32_t ndpu,
    float *data,
    float ***features_out);
float preprocessing(
    float **mean_out,
    int nfeatures,
    uint64_t npoints,
    uint64_t npadded,
    float **features,
    int_feature ***features_int_out,
    float *threshold,
    int verbose);
void postprocessing(uint64_t npoints, int nfeatures, float **features, float *mean);
float *kmeans_c(
    float **features_float,
    int_feature **features_int,
    int nfeatures,
    uint64_t npoints,
    uint64_t npadded,
    float scale_factor,
    float threshold,
    float *mean,
    int max_nclusters,
    int min_nclusters,
    int isRMSE,
    int isOutput,
    int nloops,
    int *log_iterations,
    double *log_time,
    uint32_t ndpu,
    dpu_set *allset,
    int *best_nclusters);
/**@}*/

/** @name cluster.c */
/**@{*/
void load_kernel(dpu_set *allset, const char *DPU_BINARY, uint32_t *ndpu);
void free_dpus(dpu_set *allset);
int cluster(
    uint64_t npoints,
    uint64_t npadded,
    int nfeatures,
    uint32_t ndpu,
    float **features_float,
    int_feature **features_int,
    int min_nclusters,
    int max_nclusters,
    float scale_factor,
    float threshold,
    int *best_nclusters,
    float ***cluster_centres,
    float *min_rmse,
    int isRMSE,
    int isOutput,
    int nloops,
    int *log_iterations,
    double *log_time,
    dpu_set *allset);
/**@}*/

/** @name kmeans_clustering.c */
/**@{*/
float **kmeans_clustering(
    int_feature **features_int,
    float **features_float,
    int nfeatures,
    uint64_t npoints,
    uint64_t npadded,
    unsigned int nclusters,
    int ndpu,
    float scale_factor,
    float threshold,
    int isOutput,
    uint8_t *membership,
    int *loop,
    int iteration,
    dpu_set *allset);
/**@}*/

/** @name kmeans_dpu.c */
/**@{*/
void allocateMemory(uint64_t npadded, int ndpu);
void deallocateMemory();
void populateDpu(
    int_feature **feature,
    int nfeatures,
    uint64_t npoints,
    uint64_t npadded,
    int ndpu,
    dpu_set *allset);
void kmeansDpu(
    int nfeatures,
    uint64_t npoints,
    uint64_t npadded,
    int ndpu,
    int nclusters,
    int64_t new_centers_len[ASSUMED_NR_CLUSTERS],
    int64_t new_centers[ASSUMED_NR_CLUSTERS][ASSUMED_NR_FEATURES],
    dpu_set *allset);
/**@}*/
#endif // ifndef _KMEANS_DPU_KERNEL_H_

#endif // ifndef _H_KMEANS
