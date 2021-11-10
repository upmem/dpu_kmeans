/**
 * @file kmeans_dpu.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Nanny for the DPUs.
 */

#include <dpu.h>
#include <dpu_log.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include "../kmeans.h"

static int64_t *centers_psum; /**< partial average performed by individual DPUs for each centroid */
static int **centers_pcount;  /**< partial count of points membership by each DPU */

#ifdef PERF_COUNTER
uint64_t (*counters)[HOST_COUNTERS]; /**< performance counters from every DPU */
#endif

/**
 * @brief Allocates memory for DPU communication.
 *
 * @param npadded Number of points with padding.
 * @param ndpu Number of available DPUs.
 */
void allocateMemory(Params *p)
{
    centers_psum = (int64_t *)malloc(p->ndpu * ASSUMED_NR_CLUSTERS * ASSUMED_NR_FEATURES * sizeof(*centers_psum));
    centers_pcount = (int **)malloc(p->ndpu * sizeof(*centers_pcount));
    centers_pcount[0] = (int *)malloc(p->ndpu * ASSUMED_NR_CLUSTERS * sizeof(**centers_pcount));
    for (int i = 1; i < p->ndpu; i++)
        centers_pcount[i] = centers_pcount[i - 1] + ASSUMED_NR_CLUSTERS;

#ifdef PERF_COUNTER
    counters = malloc(p->ndpu * sizeof(uint64_t[HOST_COUNTERS]));
#endif
}

/**
 * @brief Frees memory allocated by allocateMemory().
 */
void deallocateMemory()
{
    free(centers_psum);
    free(centers_pcount[0]);
    free(centers_pcount);

#ifdef PERF_COUNTER
    free(counters);
#endif
}

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
int offset(int feature, int cluster, int dpu, int nfeatures, int nclusters)
{
    return (dpu * nclusters * nfeatures) + (cluster * nfeatures) + feature;
}

/**
 * @brief Fills the DPUs with their assigned points.
 */
void populateDpu(
    Params *p,             /**< Algorithm parameters */
    int_feature **feature) /**< array: [npoints][nfeatures] */
{
    /* Iteration variables for the DPUs. */
    struct dpu_set_t dpu;
    uint32_t each_dpu;

    int *nreal_points;                     /* number of real data points on each dpu */
    int64_t remaining_points = p->npoints; /* number of yet unassigned points */

    DPU_FOREACH(p->allset, dpu, each_dpu)
    {
        int next;
        next = each_dpu * p->npointperdpu;
        DPU_ASSERT(dpu_prepare_xfer(dpu, feature[next]));
    }
    DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "t_features", 0, p->npointperdpu * p->nfeatures * sizeof(int_feature), DPU_XFER_DEFAULT));

    // telling each DPU how many real points it has to process
    nreal_points = (int *)malloc(p->ndpu * sizeof(*nreal_points));
    for (int idpu = 0; idpu < p->ndpu; idpu++)
    {
        nreal_points[idpu] = (remaining_points <= 0)                ? 0
                             : (remaining_points > p->npointperdpu) ? p->npointperdpu
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

    DPU_FOREACH(p->allset, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &nreal_points[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "npoints", 0, sizeof(int), DPU_XFER_DEFAULT));
    free(nreal_points);
}

/**
 * @brief Performs one iteration of the Lloyd algorithm on DPUs and gets the results.
 */
void kmeansDpu(
    Params *p,                                                     /**< Algorithm parameters */
    int nclusters,                                                 /**< number of clusters k */
    int64_t new_centers_len[ASSUMED_NR_CLUSTERS],                  /**< [out] number of elements in each cluster */
    int64_t new_centers[ASSUMED_NR_CLUSTERS][ASSUMED_NR_FEATURES]) /**< [out] sum of elements in each cluster */
{
    struct dpu_set_t dpu; /* Iteration variable for the DPUs. */
    uint32_t each_dpu;    /* Iteration variable for the DPUs. */

#ifdef PERF_COUNTER
    uint64_t counters_mean[HOST_COUNTERS] = {0};
#endif

    //============RUNNING ONE LLOYD ITERATION ON THE DPU==============
    DPU_ASSERT(dpu_launch(p->allset, DPU_SYNCHRONOUS));
    //================================================================

    /* DEBUG : read logs */
    // DPU_FOREACH(*allset, dpu, each_dpu) {
    //     if (each_dpu == 0)
    //         DPU_ASSERT(dpu_log_read(dpu, stdout));
    // }
    // exit(0);

    /* Performance tracking */
#ifdef PERF_COUNTER
    DPU_FOREACH(*allset, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &counters[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(*allset, DPU_XFER_FROM_DPU, "host_counters", 0, sizeof(uint64_t[HOST_COUNTERS]), DPU_XFER_DEFAULT));

    for (int icounter = 0; icounter < HOST_COUNTERS; icounter++)
    {
        int nonzero_dpus = 0;
        for (int idpu = 0; idpu < ndpu; idpu++)
            if (counters[idpu][MAIN_LOOP_CTR] != 0)
            {
                counters_mean[icounter] += counters[idpu][icounter];
                nonzero_dpus++;
            }
        counters_mean[icounter] /= nonzero_dpus;
    }
    printf("number of %s for this iteration : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[TOTAL_CTR]);
    printf("%s in main loop : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[MAIN_LOOP_CTR]);
    printf("%s in initialization : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[INIT_CTR]);
    printf("%s in critical loop arithmetic : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[CRITLOOP_ARITH_CTR]);
    printf("%s in reduction arithmetic + implicit access : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[REDUCE_ARITH_CTR]);
    printf("%s in reduction loop : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[REDUCE_LOOP_CTR]);
    printf("%s in dispatch function : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[DISPATCH_CTR]);
    printf("%s in mutexed implicit access : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[MUTEX_CTR]);

    printf("\ntotal %s in arithmetic : %ld\n", (PERF_COUNTER) ? "instructions" : "cycles", counters_mean[CRITLOOP_ARITH_CTR] + counters_mean[REDUCE_ARITH_CTR]);
    printf("percent %s in arithmetic : %.2f%%\n", (PERF_COUNTER) ? "instructions" : "cycles", 100.0 * (float)(counters_mean[CRITLOOP_ARITH_CTR] + counters_mean[REDUCE_ARITH_CTR]) / counters_mean[TOTAL_CTR]);
    printf("\n");
#endif

    /* copy back membership count per dpu (device to host) */
    DPU_FOREACH(p->allset, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &(centers_pcount[each_dpu][0])));
    }
    int nclusters_even = ((nclusters + 1) / 2) * 2;
    DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "centers_count_mram", 0, sizeof(int) * nclusters_even, DPU_XFER_DEFAULT));

    /* DEBUG : print outputed centroids counts per DPU */
    // for (int dpu_id = 0; dpu_id < ndpu; dpu_id++)
    // {
    //     for (int cluster_id = 0; cluster_id < nclusters; cluster_id++)
    //     {
    //         printf("%d ",centers_pcount[dpu_id][cluster_id]);
    //     }
    //     printf("\n");
    // }

    /* copy back centroids partial averages (device to host) */
    DPU_FOREACH(p->allset, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &centers_psum[offset(0, 0, each_dpu, p->nfeatures, nclusters)]));
    }
    DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "centers_sum_mram", 0, p->nfeatures * nclusters * sizeof(int64_t), DPU_XFER_DEFAULT));

    for (int dpu_id = 0; dpu_id < p->ndpu; dpu_id++)
    {
        for (int cluster_id = 0; cluster_id < nclusters; cluster_id++)
        {
            /* sum membership counts */
            new_centers_len[cluster_id] += centers_pcount[dpu_id][cluster_id];
            /* compute the new centroids sum */
            for (int feature_id = 0; feature_id < p->nfeatures; feature_id++)
                new_centers[cluster_id][feature_id] += centers_psum[offset(feature_id, cluster_id, dpu_id, p->nfeatures, nclusters)];
        }
    }
}
