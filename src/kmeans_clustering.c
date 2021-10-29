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

#include "kmeans.h"

static int64_t new_centers_len[ASSUMED_NR_CLUSTERS];                  /**< [nclusters]: no. of points in each cluster */
static int64_t new_centers[ASSUMED_NR_CLUSTERS][ASSUMED_NR_FEATURES]; /**< coordinates of the centroids */

/**
 * @brief Performs a final reduction of the centroids coordinates in float format.
 */
void final_reduction(
    float **feature_float,  /**< array: [npadded][nfeatures] */
    int nfeatures,          /**< number of attributes for each point */
    uint64_t npoints,       /**< number of data points */
    uint64_t npadded,       /**< number of data points with padding */
    unsigned int nclusters, /**< number of clusters in this iteration */
    int ndpu,               /**< number of available DPUs */
    uint8_t *membership,    /**< membership of each point */
    float **clusters_float, /**< [out] final centroids coordinates */
    dpu_set *allset)
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
        float **clusters_thread = (float **)malloc(nclusters * sizeof(float *));
        clusters_thread[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
        for (int i = 1; i < nclusters; i++)
            clusters_thread[i] = clusters_thread[i - 1] + nfeatures;

#pragma omp for schedule(static) collapse(2)
        for (int ipoint = 0; ipoint < npoints; ipoint++)
            for (int ifeature = 0; ifeature < nfeatures; ifeature++)
                clusters_thread[membership[ipoint]][ifeature] += feature_float[ipoint][ifeature];

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

/**
 * @brief Performs one run of the KMeans clustering algorithm.
 *
 * @return The centroids coordinates.
 */
float **kmeans_clustering(
    int_feature **features_int, /**< array: [npadded][nfeatures] */
    float **feature_float,      /**< array: [npadded][nfeatures] */
    int nfeatures,              /**< number of attributes for each point */
    uint64_t npoints,           /**< number of data points */
    uint64_t npadded,           /**< number of data points with padding */
    unsigned int nclusters,     /**< number of clusters in this iteration */
    int ndpu,                   /**< number of available DPUs */
    float threshold,            /**< loop terminating factor */
    uint8_t *membership,        /**< [out] cluster membership of each point */
    int *loop,                  /**< [out] number of inner iterations */
    int iteration,              /**< index of current outer iteration */
    dpu_set *allset)
{
    float **clusters_float;                     /* [out] final cluster coordinates */
    int_feature **clusters_int;                 /* intermediary cluster coordinates */
    float frob;                                 /* Frobenius norm of the difference in the cluster centers of two consecutive iterations */
    unsigned int npointperdpu = npadded / ndpu; /* number of points per DPU */

    /* nclusters should never be > npoints
       that would guarantee a cluster without points */
    if (nclusters > npoints)
        nclusters = npoints;

    /* making sure we are sending cluster data in multiple of 8 */
    unsigned int features_in_8bytes = 8 / sizeof(int_feature);
    int nclusters_round = ((nclusters + features_in_8bytes - 1) / features_in_8bytes) * features_in_8bytes;

    /* allocate space for and initialize returning variable clusters_int[] */
    clusters_int = (int_feature **)malloc(nclusters_round * sizeof(int_feature *));
    clusters_int[0] = (int_feature *)malloc(nclusters_round * nfeatures * sizeof(int_feature));
    for (int i = 1; i < nclusters; i++)
        clusters_int[i] = clusters_int[i - 1] + nfeatures;

    /* allocate space for and initialize returning variable clusters_float[] */
    clusters_float = (float **)malloc(nclusters * sizeof(float *));
    clusters_float[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int i = 1; i < nclusters; i++)
        clusters_float[i] = clusters_float[i - 1] + nfeatures;

    /* initialize the random clusters */
    int iteration_base_index = iteration * nclusters;
    for (int icluster = 0; icluster < nclusters; icluster++)
        for (int ifeature = 0; ifeature < nfeatures; ifeature++)
        {
            clusters_int[icluster][ifeature] = features_int[icluster + iteration_base_index][ifeature];
            clusters_float[icluster][ifeature] = clusters_int[icluster][ifeature];
        }

        /* DEBUG : print initial centroids */
        // printf("initial centroids:\n");
        // for (i = 0; i < nclusters; i++)
        // {
        //     for (j = 0; j < nfeatures; j++)
        //         printf("% d ", clusters_int[i][j]);
        //     printf("\n");
        // }
        // printf("\n");

#ifdef CPU_REDUCE
    /* initialize the membership to -1 for all */
    for (i = 0; i < npoints; i++)
        membership[i] = -1;
#endif

    /* inform DPUs of the current number of cluters */
    DPU_ASSERT(dpu_broadcast_to(*allset, "nclusters_host", 0, &nclusters, sizeof(nclusters), DPU_XFER_DEFAULT));

    /* iterate until convergence */
    do
    {
        DPU_ASSERT(dpu_broadcast_to(*allset, "c_clusters", 0, clusters_int[0], nclusters_round * nfeatures * sizeof(int_feature), DPU_XFER_DEFAULT));

        memset(new_centers, 0, sizeof(new_centers));
        memset(new_centers_len, 0, sizeof(new_centers_len));

        kmeansDpu(
            nfeatures,       /* number of attributes for each point */
            npoints,         /* number of data points */
            npadded,         /* number of data points with padding */
            ndpu,            /* number of available DPUs */
            nclusters,       /* number of clusters k */
            new_centers_len, /* [out] number of points in each cluster */
            new_centers,     /* [out] sum of points coordinates in each cluster */
            allset);

        /* DEBUG : print the centroids on each iteration */
        printf("clusters :\n");
        for (int i = 0; i < nclusters; i++)
        {
            for (int j = 0; j < nfeatures; j++)
            {
                printf("% .4f ", (float)new_centers[i][j] / new_centers_len[i]);
            }
            printf("(%d points)\n", new_centers_len[i]);
        }
        printf("\n");

        /* replace old cluster centers with new_centers */
        /* CPU side of reduction */
        frob = 0;
        // #pragma omp parallel for collapse(2) reduction(+:frob) // awful performance, don't do that
        for (int i = 0; i < nclusters; i++)
            for (int j = 0; j < nfeatures; j++)
                if (new_centers_len[i] > 0)
                {
                    double new_coordinate = (double)new_centers[i][j] / new_centers_len[i]; /* take average i.e. sum/n */
                    frob += (clusters_float[i][j] - new_coordinate) * (clusters_float[i][j] - new_coordinate);
                    clusters_float[i][j] = new_coordinate;
                    clusters_int[i][j] = lround(new_coordinate);
                }

        /* DEBUG : print convergence info */
        // printf("finished loop %d\n", loop);
        // printf("Frobenius norm  = %.12f\n", frob);
        // printf("delta = %d\n", delta);
        // printf("\n\n");

    } while (((*loop)++ < 500) && (frob > threshold)); /* makes sure loop terminates */

    printf("iterated %d times\n", *loop);

#ifdef FLT_REDUCE
    final_reduction(feature_float,
                    nfeatures,
                    npoints,
                    npadded,
                    nclusters,
                    ndpu,
                    membership,
                    clusters_float,
                    allset);
#endif

    return clusters_float;
}
