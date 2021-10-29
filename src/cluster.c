/**
 * @file cluster.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Performs the KMeans algorithm over all requested numbers of clusters.
 */

#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>

#include "kmeans.h"

static struct timeval cluster_timing; /**< Total clustering time */
static float min_rmse_ref = FLT_MAX;  /**< reference min_rmse value */

// #ifndef DPU_BINARY
// #define DPU_BINARY "src/dpu_kmeans/dpu_program/kmeans_dpu_kernel" /**< filename of the binary sent to the kernel */
// #end

/**
 * @brief Computes the lowest common multiple of two integers.
 *
 * @param n1 First integer.
 * @param n2 Second integer.
 * @return Their lowest common multiple.
 */
int get_lcm(int n1, int n2)
{
    static int max = 1;
    if (max % n1 == 0 && max % n2 == 0)
    {
        return max;
    }
    else
    {
        max++;
        get_lcm(n1, n2);
        return max;
    }
}

/**
 * @brief Computes the appropriate task size for DPU tasklets.
 *
 * @param nfeatures Number of features.
 * @param npointperdpu Number of points per DPU.
 * @return The task size in bytes.
 */
unsigned int get_task_size(int nfeatures, unsigned int npointperdpu)
{
    unsigned int task_size_in_points;
    unsigned int task_size_in_bytes;
    unsigned int task_size_in_features;

    // how many points we can fit in w_features
    unsigned int max_task_size = (WRAM_FEATURES_SIZE / sizeof(int_feature)) / nfeatures;

    // number of tasks as the smallest multiple of NR_TASKLETS higher than npointperdu / max_task_size
    unsigned int ntasks = (npointperdpu + max_task_size - 1) / max_task_size;
    ntasks = ((ntasks + NR_TASKLETS - 1) / NR_TASKLETS) * NR_TASKLETS;

    // task size has to be at least 1
    task_size_in_points = (((npointperdpu + ntasks - 1) / ntasks) < max_task_size) ? ((npointperdpu + ntasks - 1) / ntasks) : max_task_size;
    if (task_size_in_points == 0)
        task_size_in_points = 1;

    task_size_in_features = task_size_in_points * nfeatures;
    task_size_in_bytes = task_size_in_features * sizeof(int_feature);

    // task size in bytes must be a multiple of 8 for DMA alignment and of number of features x byte size of integers
    int lcm = get_lcm(sizeof(int_feature) * nfeatures, 8);
    task_size_in_bytes = (task_size_in_bytes + lcm - 1) / lcm * lcm;
    if (task_size_in_bytes > WRAM_FEATURES_SIZE)
    {
        printf("error: tasks will not fit in WRAM");
        exit(EXIT_FAILURE);
    }

    return task_size_in_bytes;
}

/**
 * @brief Performs the KMeans algorithm over all values of nclusters.
 *
 * @return Number of iterations to reach the best RMSE.
 */
int cluster(
    uint64_t npoints,           /**< number of data points */
    uint64_t npadded,           /**< number of data points with padding */
    int nfeatures,              /**< number of attributes for each point */
    uint32_t ndpu,              /**< number of available DPUs */
    float **features_float,     /**< array: [npadded][nfeatures] */
    int_feature **features_int, /**< array: [npadded][nfeatures] */
    int min_nclusters,          /**< range of min to max number of clusters */
    int max_nclusters,          /**< range of min to max number of clusters */
    float threshold,            /**< loop terminating factor */
    int *best_nclusters,        /**< [out] number between min and max with lowest RMSE */
    float ***cluster_centres,   /**< [out] [best_nclusters][nfeatures] */
    float *min_rmse,            /**< [out] minimum RMSE */
    int isRMSE,                 /**< calculate RMSE */
    int nloops,                 /**< number of iteration for each number of clusters */
    char *logname,              /**< name of the log file */
    const char *DPU_BINARY,     /**< path to the DPU kernel */
    dpu_set *allset)
{
    unsigned int nclusters;                     /* number of clusters k */
    int index = 0;                              /* number of iteration to reach the best RMSE */
    float rmse;                                 /* RMSE for each clustering */
    uint8_t *membership;                        /* which cluster a data point belongs to */
    float **tmp_cluster_centres;                /* hold coordinates of cluster centers */
    unsigned int npointperdpu = npadded / ndpu; /* number of points per DPU */

    /* parameters to calculate once here and send to the DPUs. */
    unsigned int task_size_in_points;
    unsigned int task_size_in_bytes;
    unsigned int task_size_in_features;

    /* open log file */
    FILE *fp;
    fp = fopen(logname, "w");
    fprintf(fp, "npoints : %lu, nfeatures : %d\n", npoints, nfeatures);
    fprintf(fp, "nclusters, iterations, time in seconds\n");

    printf("Possible break point\n");
    printf("loading %s\n", DPU_BINARY);
    /* load DPU binary */
    // DPU_ASSERT(dpu_load(*allset,"/home/upmemstaff/sbrocard/new_kmeans/DPU/kmeans/kmeans_dpu_kernel", NULL));
    // DPU_ASSERT(dpu_load(*allset, "src/dpu_kmeans/dpu_program/kmeans_dpu_kernel", NULL));
    DPU_ASSERT(dpu_load(*allset, DPU_BINARY, NULL));
    printf("Wasn't load\n");

    /* allocate memory for membership */
    membership = (uint8_t *)malloc(npadded * sizeof(uint8_t));

    /* =============== DPUs initialization =============== */
    /* compute the iteration variables for the DPUs */

    task_size_in_bytes = get_task_size(nfeatures, npointperdpu);

    /* realign task size in features and points */
    task_size_in_features = task_size_in_bytes / sizeof(int_feature);
    task_size_in_points = task_size_in_features / nfeatures;

    /* send computation parameters to the DPUs */
    DPU_ASSERT(dpu_broadcast_to(*allset, "nfeatures_host", 0, &nfeatures, sizeof(nfeatures), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(*allset, "npoints", 0, &npointperdpu, sizeof(npointperdpu), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(*allset, "task_size_in_points_host", 0, &task_size_in_points, sizeof(task_size_in_points), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(*allset, "task_size_in_bytes_host", 0, &task_size_in_bytes, sizeof(task_size_in_bytes), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(*allset, "task_size_in_features_host", 0, &task_size_in_features, sizeof(task_size_in_features), DPU_XFER_DEFAULT));

    printf("points per DPU : %d\n", npointperdpu);
    printf("tasks per DPU: %d\n", npointperdpu / task_size_in_points);
    printf("task size in points : %d\n", task_size_in_points);
    printf("task size in bytes : %d\n", task_size_in_bytes);

    /* fill DPUs with the data points */
    populateDpu(
        features_int, /* array: [npoints][nfeatures] */
        nfeatures,    /* number of attributes for each point */
        npoints,      /* number of real data points */
        npadded,      /* number of padded data points */
        ndpu,         /* number of available DPUs */
        allset);

    /* allocate memory for device communication */
    allocateMemory(npadded, ndpu);
    /* =============== end DPUs initialization =============== */

    printf("\nStarting calculation\n\n");

    /* sweep k from min to max_nclusters to find the best number of clusters */
    for (nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
    {
        int total_iterations = 0;

        if (nclusters > npoints)
            break; /* cannot have more clusters than points */

        cluster_timing.tv_sec = 0;
        cluster_timing.tv_usec = 0;

        /* iterate nloops times for each number of clusters */
        for (int i_init = 0; i_init < nloops; i_init++)
        {
            /* initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) */
            struct timeval tic, toc;
            int iterations_counter = 0;
            gettimeofday(&tic, NULL); // timing = omp_get_wtime();

            tmp_cluster_centres = kmeans_clustering(
                features_int,
                features_float,
                nfeatures,
                npoints,
                npadded,
                nclusters,
                ndpu,
                threshold,
                membership,
                &iterations_counter,
                i_init,
                allset);

            gettimeofday(&toc, NULL);
            cluster_timing.tv_sec += toc.tv_sec - tic.tv_sec;
            cluster_timing.tv_usec += toc.tv_usec - tic.tv_usec;

            total_iterations += iterations_counter;

            if (*cluster_centres)
            {
                free((*cluster_centres)[0]);
                free(*cluster_centres);
            }
            *cluster_centres = tmp_cluster_centres;

            /* DEBUG : print cluster centers */
            // printf("cluster centers:\n");
            // for(int icluster = 0; icluster<nclusters; icluster++)
            // {
            //     for(int ifeature = 0; ifeature<nfeatures; ifeature++)
            //     {
            //         printf("%8.4f ", (*cluster_centres)[icluster][ifeature]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");

            /* find the number of clusters with the best RMSE */
            if (isRMSE)
            {
                rmse = rms_err(
                    features_float,
                    nfeatures,
                    npoints,
                    tmp_cluster_centres,
                    nclusters);

                printf("RMSE for nclusters = %d : %f\n", nclusters, rmse);
                if (rmse < min_rmse_ref)
                {
                    min_rmse_ref = rmse;         //update reference min RMSE
                    *min_rmse = min_rmse_ref;    //update return min RMSE
                    *best_nclusters = nclusters; //update optimum number of clusters
                    index = i_init;              //update number of iteration to reach best RMSE
                }
            }
        }

        double cluster_time = ((double)(cluster_timing.tv_sec * 1000000 + cluster_timing.tv_usec)) / 1000000;
        fprintf(fp, "%9d, %10d, %15f\n", nclusters, total_iterations, cluster_time);
    }

    deallocateMemory();

    free(membership);
    fclose(fp);

    return index;
}
