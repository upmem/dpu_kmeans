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

#include "../kmeans.h"

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
 * @param p Algorithm parameters.
 * @return The task size in bytes.
 */
static unsigned int get_task_size(Params *p)
{
    unsigned int task_size_in_points;
    unsigned int task_size_in_bytes;
    unsigned int task_size_in_features;

    /* how many points we can fit in w_features */
    unsigned int max_task_size = (WRAM_FEATURES_SIZE / sizeof(int_feature)) / p->nfeatures;

    /* number of tasks as the smallest multiple of NR_TASKLETS higher than npointperdu / max_task_size */
    unsigned int ntasks = (p->npointperdpu + max_task_size - 1) / max_task_size;
    ntasks = ((ntasks + NR_TASKLETS - 1) / NR_TASKLETS) * NR_TASKLETS;

    /* task size has to be at least 1 */
    task_size_in_points = (((p->npointperdpu + ntasks - 1) / ntasks) < max_task_size)
                              ? ((p->npointperdpu + ntasks - 1) / ntasks)
                              : max_task_size;
    if (task_size_in_points == 0)
        task_size_in_points = 1;

    task_size_in_features = task_size_in_points * p->nfeatures;
    task_size_in_bytes = task_size_in_features * sizeof(int_feature);

    /* task size in bytes must be a multiple of 8 for DMA alignment and also a multiple of number of features x byte size of integers */
    int lcm = get_lcm(sizeof(int_feature) * p->nfeatures, 8);
    task_size_in_bytes = (task_size_in_bytes + lcm - 1) / lcm * lcm;
    if (task_size_in_bytes > WRAM_FEATURES_SIZE)
    {
        printf("error: tasks will not fit in WRAM");
        exit(EXIT_FAILURE);
    }

    return task_size_in_bytes;
}

/**
 * @brief Loads a binary in the DPUs.
 *
 * @param p Algorithm parameters.
 * @param DPU_BINARY path to the binary
 */
void load_kernel(Params *p, const char *DPU_BINARY)
{
    DPU_ASSERT(dpu_load(p->allset, DPU_BINARY, NULL));
}

void free_dpus(Params *p)
{
    DPU_ASSERT(dpu_free(p->allset));
}

/**
 * @brief Performs the KMeans algorithm over all values of nclusters.
 *
 * @return Number of iterations to reach the best RMSE.
 */
int cluster(
    Params *p,
    float **features_float,     /**< [in] array: [npadded][nfeatures] */
    int_feature **features_int, /**< [in] array: [npadded][nfeatures] */
    int *best_nclusters,        /**< [out] number between min and max with lowest RMSE */
    float ***cluster_centres,   /**< [out] [best_nclusters][nfeatures] */
    float *min_rmse,            /**< [out] minimum RMSE */
    int *log_iterations,        /**< [out] log of the number of iterations */
    double *log_time)           /**< [out] log of the time taken */
{
    unsigned int nclusters;        /* number of clusters k */
    unsigned int log_index = 0;    /* index of the current nclusters iteration */
    int index = 0;                 /* number of iteration to reach the best RMSE */
    float rmse;                    /* RMSE for each clustering */
    float **tmp_cluster_centres;   /* hold coordinates of cluster centers */
    float min_rmse_ref = FLT_MAX;  /* reference min_rmse value */
    struct timeval cluster_timing; /* clustering time for a given nclusters */

    /* parameters to calculate once here and send to the DPUs. */
    unsigned int task_size_in_points;
    unsigned int task_size_in_bytes;
    unsigned int task_size_in_features;

    /* =============== DPUs initialization =============== */
    /* compute the iteration variables for the DPUs */

    task_size_in_bytes = get_task_size(p);

    /* realign task size in features and points */
    task_size_in_features = task_size_in_bytes / sizeof(int_feature);
    task_size_in_points = task_size_in_features / p->nfeatures;

    /* send computation parameters to the DPUs */
    DPU_ASSERT(dpu_broadcast_to(p->allset, "nfeatures_host", 0, &p->nfeatures, sizeof(p->nfeatures), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_points_host", 0, &task_size_in_points, sizeof(task_size_in_points), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_bytes_host", 0, &task_size_in_bytes, sizeof(task_size_in_bytes), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_features_host", 0, &task_size_in_features, sizeof(task_size_in_features), DPU_XFER_DEFAULT));

    if (p->isOutput)
    {
        printf("points per DPU : %d\n", p->npointperdpu);
        printf("tasks per DPU: %d\n", p->npointperdpu / task_size_in_points);
        printf("task size in points : %d\n", task_size_in_points);
        printf("task size in bytes : %d\n", task_size_in_bytes);
    }

    /* =============== end DPUs initialization =============== */

    if (p->isOutput)
        printf("\nStarting calculation\n\n");

    /* sweep k from min to max_nclusters to find the best number of clusters */
    for (nclusters = p->min_nclusters; nclusters <= p->max_nclusters; nclusters++)
    {
        int total_iterations = 0;

        if (nclusters > p->npoints)
            break; /* cannot have more clusters than points */

        cluster_timing.tv_sec = 0;
        cluster_timing.tv_usec = 0;

        allocateClusters(p, nclusters);

        /* iterate nloops times for each number of clusters */
        for (int i_init = 0; i_init < p->nloops; i_init++)
        {
            struct timeval tic, toc;
            int iterations_counter = 0;

            gettimeofday(&tic, NULL); // `timing = omp_get_wtime();` returned absurd values

            tmp_cluster_centres = kmeans_clustering(
                p,
                features_int,
                features_float,
                nclusters,
                &iterations_counter,
                i_init);

            gettimeofday(&toc, NULL);
            cluster_timing.tv_sec += toc.tv_sec - tic.tv_sec;
            cluster_timing.tv_usec += toc.tv_usec - tic.tv_usec;

            total_iterations += iterations_counter;

            /* DEBUG : print cluster centers */
            // printf("\ncluster centers:\n");
            // for(int icluster = 0; icluster<nclusters; icluster++)
            // {
            //     for(int ifeature = 0; ifeature<nfeatures; ifeature++)
            //     {
            //         printf("%8.4f ", tmp_cluster_centres[icluster][ifeature]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");

            /* find the number of clusters with the best RMSE */
            if (p->isRMSE || p->min_nclusters != p->max_nclusters || p->nloops > 1)
            {
                rmse = rms_err(
                    p,
                    features_float,
                    tmp_cluster_centres,
                    nclusters);

                if (p->isOutput)
                    printf("RMSE for nclusters = %d : %f\n", nclusters, rmse);

                if (rmse < min_rmse_ref)
                {
                    min_rmse_ref = rmse;         // update reference min RMSE
                    *min_rmse = min_rmse_ref;    // update return min RMSE
                    *best_nclusters = nclusters; // update optimum number of clusters
                    index = i_init;              // update number of iteration to reach best RMSE
                    /* update best cluster centres */
                    if (*cluster_centres)
                    {
                        free((*cluster_centres)[0]);
                        free(*cluster_centres);
                    }
                    *cluster_centres = tmp_cluster_centres;
                }
            }
            else
            {
                if (*cluster_centres)
                {
                    free((*cluster_centres)[0]);
                    free(*cluster_centres);
                }
                *cluster_centres = tmp_cluster_centres;
            }
        }

        deallocateClusters();

        /* logging number of iterations and time taken */
        double cluster_time = ((double)(cluster_timing.tv_sec * 1000000 + cluster_timing.tv_usec)) / 1000000;
        log_iterations[log_index] = total_iterations;
        log_time[log_index] = cluster_time;
        log_index++;
    }

    /* DEBUG: print best clusters */
    // printf("best nclusters: %d\n", *best_nclusters);
    // printf("trying\n");
    // for (int icluster = 0; icluster < *best_nclusters; icluster++)
    // {
    //     for (int ifeature = 0; ifeature < nfeatures; ifeature++)
    //         printf("%f ",(*cluster_centres)[icluster][ifeature]);
    //     printf("\n");
    // }

    return index;
}
