/**
 * @file kmeans_clustering.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Performs one run of the KMeans clustering algorithm.
 */

#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#include "../kmeans.h"

static int64_t new_centers_len[ASSUMED_NR_CLUSTERS];                  /**< [nclusters]: no. of points in each cluster */
static int64_t new_centers[ASSUMED_NR_CLUSTERS][ASSUMED_NR_FEATURES]; /**< coordinates of the centroids */
static float **clusters_float;                                        /**< final cluster coordinates */
static float **old_clusters_float;                                    /**< previous cluster coordinates */
static int_feature **clusters_int;                                    /**< integer cluster coordinates */

void allocateClusters(Params *p, unsigned int nclusters)
{
    /* making sure we are sending cluster data in multiple of 8 */
    unsigned int features_in_8bytes = 8 / sizeof(int_feature);
    int nclusters_round = ((nclusters + features_in_8bytes - 1) / features_in_8bytes) * features_in_8bytes;

    /* allocate space for and initialize exchange variable clusters_int[] */
    clusters_int = (int_feature **)malloc(nclusters_round * sizeof(*clusters_int));
    clusters_int[0] = (int_feature *)malloc(nclusters_round * p->nfeatures * sizeof(**clusters_int));
    for (int i = 1; i < nclusters; i++)
        clusters_int[i] = clusters_int[i - 1] + p->nfeatures;

    /* allocate space for and initialize temporary variable old_clusters_float[] */
    old_clusters_float = (float **)malloc(nclusters * sizeof(*old_clusters_float));
    old_clusters_float[0] = (float *)malloc(nclusters * p->nfeatures * sizeof(**old_clusters_float));
    for (int i = 1; i < nclusters; i++)
        old_clusters_float[i] = old_clusters_float[i - 1] + p->nfeatures;
}

void deallocateClusters()
{
    free(old_clusters_float[0]);
    free(old_clusters_float);
    free(clusters_int[0]);
    free(clusters_int);
}

#ifdef FLT_REDUCE
static uint8_t *membership;

/**
 * @brief Allocates memory for membership table.
 *
 * @param p Algorithm parameters.
 */
void allocateMembershipTable(Params *p)
{
    membership = (uint8_t *)malloc(p->npadded * sizeof(*membership));
}

/**
 * @brief Deallocates membership table.
 *
 */
void deallocateMembershipTable()
{
    free(membership);
}

/**
 * @brief Performs a final reduction of the centroids coordinates in float format.
 */
void final_reduction(
    float **features_float, /**< array: [npadded][nfeatures] */
    int nfeatures,          /**< number of attributes for each point */
    uint64_t npoints,       /**< number of data points */
    uint64_t npadded,       /**< number of data points with padding */
    unsigned int nclusters, /**< number of clusters in this iteration */
    int ndpu,               /**< number of available DPUs */
    float **clusters_float, /**< [out] final centroids coordinates */
    dpu_set *allset)        /**< pointer to the set of all assigned DPUs */
{
    uint32_t each_dpu;    /* Iteration variable for the DPUs. */
    struct dpu_set_t dpu; /* Iteration variable for the DPUs. */

    unsigned int npointperdpu = npadded / ndpu; /**< number of points per DPU */

#ifdef PERF_COUNTER
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    printf("\nPerformance of final CPU-side reduction:\n");
#endif

    /* copy back points membership */
    DPU_FOREACH(*allset, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &(membership[each_dpu * npointperdpu])));
    }
    DPU_ASSERT(dpu_push_xfer(*allset, DPU_XFER_FROM_DPU, "t_membership", 0, npointperdpu * sizeof(*membership), DPU_XFER_DEFAULT));

#ifdef PERF_COUNTER
    gettimeofday(&toc, NULL);
    printf("membership transfer time: %f seconds\n", time_seconds(tic, toc));
    gettimeofday(&tic, NULL);
#endif

    memset(clusters_float[0], 0, sizeof(**clusters_float) * nclusters * nfeatures);

#pragma omp parallel
    {
        float **clusters_thread = (float **)malloc(nclusters * sizeof(*clusters_thread));
        clusters_thread[0] = (float *)calloc(nclusters * nfeatures, sizeof(**clusters_thread));
        for (int i = 1; i < nclusters; i++)
            clusters_thread[i] = clusters_thread[i - 1] + nfeatures;

#pragma omp for schedule(static) collapse(2)
        for (int ipoint = 0; ipoint < npoints; ipoint++)
            for (int ifeature = 0; ifeature < nfeatures; ifeature++)
                clusters_thread[membership[ipoint]][ifeature] += features_float[ipoint][ifeature];

#pragma omp critical
        for (int icluster = 0; icluster < nclusters; icluster++)
            for (int ifeature = 0; ifeature < nfeatures; ifeature++)
                clusters_float[icluster][ifeature] += clusters_thread[icluster][ifeature];

        free(clusters_thread[0]);
        free(clusters_thread);
    }

    for (int ifeature = 0; ifeature < nfeatures; ifeature++)
        for (int icluster = 0; icluster < nclusters; icluster++)
            clusters_float[icluster][ifeature] /= new_centers_len[icluster];

#ifdef PERF_COUNTER
    gettimeofday(&toc, NULL);
    printf("final reduction time: %f seconds\n\n", time_seconds(tic, toc));
#endif
}
#endif

/**
 * @brief Performs one run of the KMeans clustering algorithm.
 *
 * @return The centroids coordinates.
 */
float **kmeans_clustering(
    Params *p,                  /**< Algorithm parameters.*/
    int_feature **features_int, /**< array: [npadded][nfeatures] */
    float **features_float,     /**< array: [npadded][nfeatures] */
    unsigned int nclusters,     /**< number of clusters in this iteration */
    int *loop,                  /**< [out] number of inner iterations */
    int i_init)                 /**< index of current outer iteration */
{
    float frob; /* Frobenius norm of the difference in the cluster
                   centers of two consecutive iterations */

    float **switch_clusters_float; /* pointer for switching */

    /* nclusters should never be > npoints
       that would guarantee a cluster without points */
    if (nclusters > p->npoints)
        nclusters = p->npoints;

    /* making sure we are sending cluster data in multiple of 8 */
    unsigned int features_in_8bytes = 8 / sizeof(int_feature);
    int nclusters_round = ((nclusters + features_in_8bytes - 1) / features_in_8bytes) * features_in_8bytes;

    /* allocate space for and initialize returning variable clusters_float[] */
    clusters_float = (float **)malloc(nclusters * sizeof(*clusters_float));
    clusters_float[0] = (float *)malloc(nclusters * p->nfeatures * sizeof(**clusters_float));
    for (int i = 1; i < nclusters; i++)
        clusters_float[i] = clusters_float[i - 1] + p->nfeatures;

    /* initialize the random clusters */
    int iteration_base_index = i_init * nclusters;
    for (int icluster = 0; icluster < nclusters; icluster++)
        for (int ifeature = 0; ifeature < p->nfeatures; ifeature++)
        {
            clusters_int[icluster][ifeature] = features_int[(icluster + iteration_base_index) % p->npoints][ifeature];
            clusters_float[icluster][ifeature] = clusters_int[icluster][ifeature];
        }

    /* DEBUG : print initial centroids */
    // printf("initial centroids:\n");
    // for (int icluster = 0; icluster < nclusters; icluster++)
    // {
    //     for (int ifeature = 0; ifeature < nfeatures; ifeature++)
    //         printf("% d ", clusters_int[icluster][ifeature]);
    //     printf("\n");
    // }
    // printf("\n");

#ifdef FLT_REDUCE
    /* initialize the membership to -1 for all */
    for (i = 0; i < npoints; i++)
        membership[i] = -1;
#endif

    /* inform DPUs of the current number of cluters */
    DPU_ASSERT(dpu_broadcast_to(p->allset, "nclusters_host", 0, &nclusters, sizeof(nclusters), DPU_XFER_DEFAULT));

    /* iterate until convergence */
    do
    {
        DPU_ASSERT(dpu_broadcast_to(p->allset, "c_clusters", 0, clusters_int[0], nclusters_round * p->nfeatures * sizeof(int_feature), DPU_XFER_DEFAULT));

        memset(new_centers, 0, sizeof(new_centers));
        memset(new_centers_len, 0, sizeof(new_centers_len));

        kmeansDpu(
            p,
            nclusters,       /* number of clusters k */
            new_centers_len, /* [out] number of points in each cluster */
            new_centers);     /* [out] sum of points coordinates in each cluster */

        /* DEBUG : print the centroids on each iteration */
        // printf("clusters :\n");
        // for (int i = 0; i < nclusters; i++)
        // {
        //     for (int j = 0; j < nfeatures; j++)
        //     {
        //         printf("% .4f ", (float)new_centers[i][j] / new_centers_len[i]);
        //     }
        //     printf("(%d points)\n", new_centers_len[i]);
        // }
        // printf("\n");

        /* CPU side of reduction */

        /* switch old and new cluster pointers */
        if (*loop != 0)
        {
            switch_clusters_float = old_clusters_float;
            old_clusters_float = clusters_float;
            clusters_float = switch_clusters_float;
        }

        /* replace old cluster centers with new_centers  */
        // #pragma omp parallel for collapse(2) reduction(+:frob) // awful performance, don't do that
        for (int i = 0; i < nclusters; i++)
            for (int j = 0; j < p->nfeatures; j++)
                if (new_centers_len[i] > 0)
                {
                    float new_coordinate = (float)new_centers[i][j] / new_centers_len[i]; /* take average i.e. sum/n */
                    clusters_int[i][j] = lround(new_coordinate);
                    new_coordinate /= p->scale_factor;
                    clusters_float[i][j] = new_coordinate;
                }
                else
                    clusters_float[i][j] = old_clusters_float[i][j];

        /* compute Frobenius norm */
        frob = 0;
        if(*loop != 0)
        {
            for (int i = 0; i < nclusters; i++)
                for (int j = 0; j < p->nfeatures; j++)
                {
                    float diff = clusters_float[i][j] - old_clusters_float[i][j];
                    frob += diff * diff;
                }
        }

        /* DEBUG : print convergence info */
        // printf("finished loop %d\n", *loop);
        // printf("Frobenius norm  = %.12f\n", frob);
        // printf("threshold = %f\n", p->threshold);
        // printf("\n");

        (*loop)++;
    } while (((*loop < p->max_iter) && (frob > p->threshold)) || *loop == 1); /* makes sure loop terminates */

    if (p->isOutput)
        printf("iterated %d times\n", *loop);

#ifdef FLT_REDUCE
    final_reduction(features_float,
                    nfeatures,
                    npoints,
                    npadded,
                    nclusters,
                    ndpu,
                    clusters_float,
                    allset);
#endif

    return clusters_float;
}
