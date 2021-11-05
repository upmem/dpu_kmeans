/**
 * @file kmeans.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Main file for the KMeans algorithm.
 */

#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include <getopt.h>
#include <libgen.h>
#include <sys/time.h>

#include "kmeans.h"

/**
 * @brief Returns the seconds elapsed between two timeval structures.
 *
 * @param tic [in] First timeval.
 * @param toc [in] Second timeval.
 * @return double Elapsed time in seconds.
 */
double time_seconds(struct timeval tic, struct timeval toc)
{
    struct timeval timing;
    timing.tv_sec += toc.tv_sec - tic.tv_sec;
    timing.tv_usec += toc.tv_usec - tic.tv_usec;
    double time = ((double)(timing.tv_sec * 1000000 + timing.tv_usec)) / 1000000;

    return time;
}

/**
 * @brief Removes the extension from a file name.
 *
 * @param fname [in] The file name string.
 */
static void strip_ext(char *fname)
{
    char *end = fname + strlen(fname);

    while (end > fname && *end != '.')
        --end;

    if (end > fname)
        *end = '\0';
}

/**
 * @brief Reads a binary input file from disk.
 */
void read_bin_input(
    const char *filename,  /**< [in] The file name. */
    uint64_t *npoints_out, /**< [out] Number of points. */
    uint64_t *npadded_out, /**< [out] Number of points with padding added. */
    int *nfeatures_out,    /**< [out] Number of features. */
    uint32_t ndpu,         /**< [in] Number of available DPUs. */
    float ***features_out) /**< [out] Vector of features. */
{
    uint64_t npoints, npadded;
    int nfeatures;
    float **features;

    FILE *infile;
    if ((infile = fopen(filename, "rb")) == NULL)
    {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
    }

    /* get nfeatures and npoints */
    fread(&npoints, sizeof(uint64_t), 1, infile);
    fread(&nfeatures, sizeof(int), 1, infile);

    /* rounding the size of the input to the smallest multiple of 8*ndpu larger than npoints */
    npadded = ((npoints + 8 * ndpu - 1) / (8 * ndpu)) * 8 * ndpu;

    /* allocate space for features[][] and read attributes of all objects */
    features = (float **)malloc(npadded * sizeof(float *));
    features[0] = (float *)malloc(npadded * nfeatures * sizeof(float));
    for (int ipoint = 1; ipoint < npadded; ipoint++)
        features[ipoint] = features[ipoint - 1] + nfeatures;

    /* checking that we managed to assign enough memory */
    if (!features[0])
    {
        perror("malloc features[0]");
        exit(EXIT_FAILURE);
    }

    fread(features[0], sizeof(float), npoints * nfeatures, infile);

    fclose(infile);

    *features_out = features;
    *npoints_out = npoints;
    *npadded_out = npadded;
    *nfeatures_out = nfeatures;
}

/**
 * @brief Reads a text input file from disk.
 */
void read_txt_input(
    const char *filename,  /**< [in] The file name. */
    uint64_t *npoints_out, /**< [out] Number of points. */
    uint64_t *npadded_out, /**< [out] Number of points with padding added. */
    int *nfeatures_out,    /**< [out] Number of features. */
    uint32_t ndpu,         /**< [in] Number of available DPUs. */
    float ***features_out) /**< [out] Vector of features. */
{
    char line[1024];
    uint64_t npoints = 0;
    uint64_t npadded;
    int nfeatures = 0;
    float **features;

    FILE *infile;
    if ((infile = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
        if (strtok(line, " \t\n") != 0)
            npoints++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL)
    {
        if (strtok(line, " \t\n") != 0)
        {
            /* ignore the id (first attribute): nfeatures = 1; */
            while (strtok(NULL, " ,\t\n") != NULL)
                nfeatures++;
            break;
        }
    }
    /* rounding the size of the input to the smallest multiple of 8*ndpu larger than npoints */
    npadded = ((npoints + 8 * ndpu - 1) / (8 * ndpu)) * 8 * ndpu;

    /* allocate space for features[] and read attributes of all objects */
    features = (float **)malloc(npadded * sizeof(float *));
    features[0] = (float *)malloc(npadded * nfeatures * sizeof(float));
    for (int ipoint = 1; ipoint < npadded; ipoint++)
        features[ipoint] = features[ipoint - 1] + nfeatures;

    /* checking that we managed to assign enough memory */
    if (!features[0])
    {
        perror("malloc features[0]");
        exit(EXIT_FAILURE);
    }

    rewind(infile);
    {
        int ifeature_global = 0;
        while (fgets(line, 1024, infile) != NULL)
        {
            // if(i%10==0)
            //     printf("line %d\n", i);
            if (strtok(line, " \t\n") == NULL)
                continue;
            for (int ifeature = 0; ifeature < nfeatures; ifeature++)
            {
                features[0][ifeature_global] = atof(strtok(NULL, " ,\t\n"));
                ifeature_global++;
            }
        }
    }
    fclose(infile);

    *features_out = features;
    *npoints_out = npoints;
    *npadded_out = npadded;
    *nfeatures_out = nfeatures;
}

/**
 * @brief Saves the input data in a binary file for faster access next time.
 *
 * @param filename_in [in] Name of the input text file.
 * @param npoints [in] Number of points.
 * @param nfeatures [in] Number of features.
 * @param features [npoints][nfeatures] Feature array.
 */
void save_dat_file(const char *filename_in, uint64_t npoints, int nfeatures, float **features)
{
    char *filename = strdup(filename_in);
    char suffix[] = ".dat";

    int n = strlen(filename) + strlen(suffix);
    char *dat_name = (char *)malloc(n * sizeof(char));

    strcpy(dat_name, filename);
    strip_ext(dat_name);
    strcat(dat_name, ".dat");

    printf("Writing points in binary format to %s\n", dat_name);

    FILE *binfile;
    binfile = fopen(dat_name, "wb");
    fwrite(&npoints, sizeof(uint64_t), 1, binfile);
    fwrite(&nfeatures, sizeof(int), 1, binfile);
    fwrite(features[0], sizeof(float), npoints * nfeatures, binfile);
    fclose(binfile);

    free(filename);
    free(dat_name);
}

/**
 * @brief Formats a flat array into a bidimensional representation
 */
void format_array_input(
    uint64_t npoints,      /**< [in] Number of points. */
    uint64_t *npadded_out, /**< [out] Number of points with padding. */
    int nfeatures,         /**< [in] Number of features. */
    uint32_t ndpu,         /**< [in] Number of available DPUs */
    float *data,           /**< [in] The data as a flat table */
    float ***features_out) /**< [out] The data as two dimensional table */
{
    uint64_t npadded;
    npadded = ((npoints + 8 * ndpu - 1) / (8 * ndpu)) * 8 * ndpu;

    float **features = (float **)malloc(npadded * sizeof(float *));
    features[0] = data;
    for (int ipoint = 1; ipoint < npadded; ipoint++)
        features[ipoint] = features[ipoint - 1] + nfeatures;

    *npadded_out = npadded;
    *features_out = features;
}

/**
 * @brief Preprocesses the data before running the KMeans algorithm.
 *
 * @return float Scaling factor applied to the input data.
 */
float preprocessing(
    float **mean_out,                /**< [out] Per-feature average. */
    int nfeatures,                   /**< [in] Number of features. */
    uint64_t npoints,                /**< [in] Number of points. */
    uint64_t npadded,                /**< [in] Number of points with padding. */
    float **features,                /**< [in] Features as floats. */
    int_feature ***features_int_out, /**< [out] Features as integers. */
    float *threshold,                /**< [in,out] Termination criterion for the algorithm. */
    int verbose)                     /**< [in] Whether or not to print runtime information. */
{
    uint64_t ipoint;
    int ifeature;

    float *mean;
    double *variance;
    int_feature **features_int;
    double avg_variance;
    float max_feature = 0;
    float scale_factor;

#ifdef PERF_COUNTER
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
#endif

    /* DEBUG : print features head */
    // printf("features head:\n");
    // for (int ipoint = 0; ipoint < 10; ipoint++)
    // {
    //     for (int ifeature = 0; ifeature < nfeatures; ifeature++)
    //         printf("%.4f ", features[ipoint][ifeature]);
    //     printf("\n");
    // }
    // printf("\n");

    mean = (float *)calloc(nfeatures, sizeof(float));
    variance = (double *)calloc(nfeatures, sizeof(double));
/* compute mean by feature */
#pragma omp parallel for collapse(2) \
    reduction(+                      \
              : mean[:nfeatures])
    for (ifeature = 0; ifeature < nfeatures; ifeature++)
        for (ipoint = 0; ipoint < npoints; ipoint++)
            mean[ifeature] += features[ipoint][ifeature];

#pragma omp parallel for
    for (ifeature = 0; ifeature < nfeatures; ifeature++)
        mean[ifeature] /= npoints;

    /* DEBUG : print the means per feature */
    // printf("means = ");
    // for (ifeature = 0; ifeature < nfeatures; ifeature++)
    //     printf(" %.4f",mean[ifeature]);
    // printf("\n");

    /* subtract mean from each feature */
#pragma omp parallel for collapse(2)
    for (ipoint = 0; ipoint < npoints; ipoint++)
        for (ifeature = 0; ifeature < nfeatures; ifeature++)
            features[ipoint][ifeature] -= mean[ifeature];

    /* ****** discretization ****** */

    /* get maximum absolute value of features */
#pragma omp parallel for collapse(2) \
    reduction(max                    \
              : max_feature)
    for (ipoint = 0; ipoint < npoints; ipoint++)
        for (ifeature = 0; ifeature < nfeatures; ifeature++)
            if (fabsf(features[ipoint][ifeature]) > max_feature)
                max_feature = fabsf(features[ipoint][ifeature]);
    switch (sizeof(int_feature))
    {
    case 1UL:
        scale_factor = INT8_MAX / max_feature / 2;
        break;
    case 2UL:
        scale_factor = INT16_MAX / max_feature / 2;
        break;
    case 4UL:
        scale_factor = INT32_MAX / max_feature / 2;
        break;
    default:
        printf("Error: unsupported type for int_feature.\n");
        exit(0);
    }

    if (verbose)
    {
        printf("max absolute value : %f\n", max_feature);
        printf("scale factor = %.4f\n", scale_factor);
    }

    /* allocate space for features_int[][] and convert attributes of all objects */
    features_int = (int_feature **)malloc(npadded * sizeof(int_feature *));
    features_int[0] = (int_feature *)malloc(npadded * nfeatures * sizeof(int_feature));
    for (ipoint = 1; ipoint < npadded; ipoint++)
        features_int[ipoint] = features_int[ipoint - 1] + nfeatures;

    /* checking that we managed to assign enough memory */
    if (!features_int[0])
    {
        perror("malloc features_int[0]");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for collapse(2)
    for (ipoint = 0; ipoint < npoints; ipoint++)
        for (ifeature = 0; ifeature < nfeatures; ifeature++)
            features_int[ipoint][ifeature] = lroundf(features[ipoint][ifeature] * scale_factor);

    /* DEBUG : print features head */
    // printf("features head:\n");
    // for (int ipoint = 0; ipoint < 10; ipoint++)
    // {
    //     for (int ifeature = 0; ifeature < nfeatures; ifeature++)
    //         printf("%8d ", features_int[ipoint][ifeature]);
    //     printf("\n");
    // }
    // printf("\n");

    /* DEBUG : print features maxes */
    // printf("features max:\n");
    // for (ifeature = 0; ifeature < nfeatures; ifeature++){
    //     int max_features_int = 0;
    //     for (ipoint = 0; ipoint < npoints; ipoint++){
    //         if (features_int[ipoint][ifeature] > max_features_int)
    //             max_features_int = features_int[ipoint][ifeature];
    //     }
    //     printf("%d ", max_features_int);
    // }
    // printf("\n");

    /* ***** discretization end ***** */

    /* compute variance by feature */
#pragma omp parallel for collapse(2) \
    reduction(+                      \
              : variance[:nfeatures])
    for (ipoint = 0; ipoint < npoints; ipoint++)
        for (ifeature = 0; ifeature < nfeatures; ifeature++)
            variance[ifeature] += (double)features_int[ipoint][ifeature] * features_int[ipoint][ifeature];

#pragma omp parallel for
    for (ifeature = 0; ifeature < nfeatures; ifeature++)
        variance[ifeature] /= npoints;

    /* compute average of variance */
    avg_variance = 0;
#pragma omp parallel for reduction(+ \
                                   : avg_variance)
    for (ifeature = 0; ifeature < nfeatures; ifeature++)
        avg_variance += variance[ifeature];
    avg_variance /= nfeatures;
    *threshold *= avg_variance;

#ifdef PERF_COUNTER
    /* compute time spent on preprocessing */
    gettimeofday(&toc, NULL);
    printf("preprocessing time: %f seconds\n\n", time_seconds(tic, toc));
#endif

    if (verbose)
    {
        printf("avg_variance = %.4f\n", avg_variance);
        printf("threshold = %.4f\n", *threshold);
        printf("\npreprocessing completed\n\n");
    }

    free(variance);

    /* DEBUG */
    // printf("means:");
    // for(i = 0; i< nfeatures; i++)
    // {
    //     printf("%d ", mean[i]);
    // }
    // printf("\n");
    // for(i = 0; i < 5; i++)
    // {
    //     for(j = 0; j< nfeatures; j++)
    //     {
    //         printf("%d ", features_int[i][j]);
    //     }
    //     printf("\n");
    // }

    *mean_out = mean;
    *features_int_out = features_int;

    return scale_factor;
}

/**
 * @brief Restores the input data to its original state.
 *
 * @param npoints [in] number of points
 * @param nfeatures [in] number of features
 * @param features [in,out] array of features
 * @param mean [in] average computed during preprocessing
 */
void postprocessing(uint64_t npoints, int nfeatures, float **features, float *mean)
{
#pragma omp parallel for collapse(2)
    for (uint64_t ipoint = 0; ipoint < npoints; ipoint++)
        for (int ifeature = 0; ifeature < nfeatures; ifeature++)
            features[ipoint][ifeature] += mean[ifeature];
}

/**
 * @brief Checks for errors in the input
 *
 * @param npoints [in] Number of points.
 * @param npadded [in] Number of points with padding.
 * @param min_nclusters [in] Minimum number of clusters.
 * @param max_nclusters [in] Maximum number of clusters.
 * @param nfeatures [in] Number of features.
 * @param ndpu [in] Number of available DPUs.
 */
static void error_check(uint64_t npoints, uint64_t npadded, int min_nclusters, int max_nclusters, int nfeatures, uint32_t ndpu)
{
    if (npoints < min_nclusters)
    {
        printf("Error: min_nclusters(%d) > npoints(%lu) -- cannot proceed\n", min_nclusters, npoints);
        exit(EXIT_FAILURE);
    }
    if ((max_nclusters < min_nclusters) || (max_nclusters > ASSUMED_NR_CLUSTERS))
    {
        printf("Error: min_nclusters(%d) > max_nclusters(%lu) or max_nclusters > max clusters allowed(%d) -- cannot proceed\n", min_nclusters, npoints, ASSUMED_NR_CLUSTERS);
        exit(EXIT_FAILURE);
    }
    if (ASSUMED_NR_FEATURES < nfeatures)
    {
        printf("Error: nfeatures(%d) > max clusters allowed(%d) -- cannot proceed\n", nfeatures, ASSUMED_NR_FEATURES);
        exit(EXIT_FAILURE);
    }
    if (npadded * nfeatures / ndpu > MAX_FEATURE_DPU)
    {
        printf("Error: npadded*nfeatures/ndpu(%lu) > max features allowed per dpu(%d) -- cannot proceed\n", npadded * nfeatures / ndpu, MAX_FEATURE_DPU);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Output to array.
 *
 * @param best_nclusters [in] Best number of clusters according to RMSE.
 * @param nfeatures [in] Number of features.
 * @param cluster_centres [in] Coordinate of clusters centres for the best iteration.
 * @param scale_factor [in] Factor that was used in quantizing the data.
 * @param mean [in] Mean that was subtracted during preprocessing.
 * @return float* The return array
 */
static float *array_output(int best_nclusters, int nfeatures, float **cluster_centres, float scale_factor, float *mean)
{
    float *output_clusters = (float *)malloc(best_nclusters * nfeatures * sizeof(float));
    for (int icluster = 0; icluster < best_nclusters; icluster++)
        for (int ifeature = 0; ifeature < nfeatures; ifeature++)
        {
#ifdef FLT_REDUCE
            output_clusters[ifeature + icluster * nfeatures] = cluster_centres[icluster][ifeature] + mean[ifeature];
#else
            output_clusters[ifeature + icluster * nfeatures] = cluster_centres[icluster][ifeature] / scale_factor + mean[ifeature];
#endif
        }
    return output_clusters;
}

/**
 * @brief output to the command line
 *
 */
static void cli_output(
    int min_nclusters,       /**< [in] lower bound of the number of clusters */
    int max_nclusters,       /**< [in] upper bound of the number of clusters */
    int nfeatures,           /**< [in] number of features */
    float **cluster_centres, /**< [in] coordinate of clusters centres for the best iteration */
    float scale_factor,      /**< [in] factor that was used in quantizing the data */
    float *mean,             /**< [in] mean that was subtracted during preprocessing */
    int nloops,              /**< [in] how many times the algorithm will be executed for each number of clusters */
    int isRMSE,              /**< [in] whether or not RMSE is computed */
    float rmse,              /**< [in] value of the RMSE for the best iteration */
    int index)               /**< [in] number of trials for the best RMSE */
{
    /* cluster center coordinates
       :displayed only for when k=1*/
    if (min_nclusters == max_nclusters)
    {
        printf("\n================= Centroid Coordinates =================\n");
        for (int icluster = 0; icluster < max_nclusters; icluster++)
        {
            printf("%2d:", icluster);
            for (int ifeature = 0; ifeature < nfeatures; ifeature++)
#ifdef FLT_REDUCE
                printf(" % 10.6f", cluster_centres[icluster][ifeature] + mean[ifeature]);
#else
                printf(" % 10.6f", cluster_centres[icluster][ifeature] / scale_factor + mean[ifeature]);
#endif
            printf("\n");
        }
    }

    printf("Number of Iteration: %d\n", nloops);

    if (min_nclusters == max_nclusters && isRMSE)
    {
        if (nloops != 1)
        { // single k, multiple iteration
            printf("Number of trials to approach the best RMSE of %.3f is %d\n", rmse, index + 1);
        }
        else
        { // single k, single iteration
            printf("Root Mean Squared Error: %.3f\n", rmse);
        }
    }
}

/**
 * @brief Main function for the KMeans algorithm.
 *
 * @return float* The centroids coordinates found by the algorithm.
 */
float *kmeans_c(
    float **features_float,     /**< [in] array of features  */
    int_feature **features_int, /**< [in] array of quantized features */
    int nfeatures,              /**< [in] number of feature */
    uint64_t npoints,           /**< [in] number of points */
    uint64_t npadded,           /**< [in] number of points with padding */
    float scale_factor,         /**< [in] scale factor used in quantizaton */
    float threshold,            /**< [in] threshold for termination of the algorithm */
    float *mean,                /**< [in] feature-wise mean subtracted during preprocessing */
    int max_nclusters,          /**< [in] upper bound of the number of clusters */
    int min_nclusters,          /**< [in] lower bound of the number of clusters */
    int isRMSE,                 /**< [in] whether or not RMSE is computed */
    int isOutput,               /**< [in] whether or not to print the centroids and runtime information */
    int nloops,                 /**< [in] how many times the algorithm will be executed for each number of clusters */
    int *log_iterations,        /**< [out] Number of iterations per nclusters */
    double *log_time,           /**< [out] Time taken per nclusters */
    uint32_t ndpu,              /**< [in] number of assigned DPUs */
    dpu_set *allset,            /**< [in] set of all assigned DPUs */
    int *best_nclusters)        /**< [out] best number of clusters according to RMSE */
{
    /* Variables for I/O. */
    float *output_clusters; /* return pointer */

    // dpu_set allset; /* Set of all available DPUs. */

    /* Data arrays. */
    float **cluster_centres = NULL; /* array of centroid coordinates */
    // float *mean;                    /* feature-wise average of points coordinates */
    // int_feature **cluster_centres_int = NULL; /* array of discretized centroid coordinates */

    /* Generated values. */
    int index;  /* number of iterations on the best run */
    float rmse; /* RMSE value */
    // int best_nclusters = 0; /* best number of clusters according to RMSE */

    /* ============== DPUs init ==============*/
    /* necessary to do it first to know the n° of available DPUs */
    // TO REMOVE: refactored to Container class
    // DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &allset));
    // DPU_ASSERT(dpu_get_nr_dpus(allset, &ndpu));
    /* ============== DPUs init end ==========*/

    /* ============== I/O begin ==============*/
    // TO REMOVE: refactored to Container class
    // if (fileInput)
    // {
    //     if (isBinaryFile)
    //     { //Binary file input
    //         read_bin_input(filename, &npoints, &npadded, &nfeatures, ndpu, &features_float);
    //     }
    //     else
    //     { //Text file input
    //         read_txt_input(filename, &npoints, &npadded, &nfeatures, ndpu, &features_float);

    //         /* Saving features as a binary for next time */
    //         save_dat_file(filename, npoints, nfeatures, features_float);
    //     }
    // }
    // else
    // {
    //     format_array_input(npoints, &npadded, nfeatures, ndpu, data, &features_float);
    // }

    if (isOutput)
    {
        printf("\nNumber of objects without padding: %lu\n", npoints);
        printf("Number of objects with padding: %lu\n", npadded);
        printf("Number of features: %d\n", nfeatures);
        printf("Number of DPUs: %d\n", ndpu);
    }
    /* ============== I/O end ==============*/

    // error check for clusters
    error_check(npoints, npadded, min_nclusters, max_nclusters, nfeatures, ndpu);

    /* ======================= pre-processing ===========================*/
    // TO REMOVE: refactored to Container class
    // scale_factor = preprocessing(&mean, nfeatures, npoints, npadded, features_float, &features_int, &threshold);
    /* ======================= pre-processing end =======================*/

    /* ======================= core of the clustering ===================*/

    cluster_centres = NULL;
    index = cluster(npoints,          /* number of data points */
                    npadded,          /* number of data points with padding */
                    nfeatures,        /* number of features for each point */
                    ndpu,             /* number of available DPUs */
                    features_float,   /* array: [npoints][nfeatures] */
                    features_int,     /* array: [npoints][nfeatures] */
                    min_nclusters,    /* range of min to max number of clusters */
                    max_nclusters,    /* range of min to max number of clusters */
                    threshold,        /* loop termination factor */
                    best_nclusters,   /* [out] number between min and max */
                    &cluster_centres, /* [out] [best_nclusters][nfeatures] */
                    &rmse,            /* Root Mean Squared Error */
                    isRMSE,           /* calculate RMSE */
                    isOutput,         /* whether or not to print runtime information */
                    nloops,           /* number of iteration for each number of clusters */
                    log_iterations,   /* log of the number of iterations */
                    log_time,         /* log of the time taken */
                    allset);          /* set of all DPUs */

    /* =============== Array Output ====================== */

    output_clusters = array_output(*best_nclusters, nfeatures, cluster_centres, scale_factor, mean);

    /* =============== Command Line Output =============== */

    if (isOutput)
        cli_output(min_nclusters, max_nclusters, nfeatures, cluster_centres, scale_factor, mean, nloops, isRMSE, rmse, index);

    /* =============== Postprocessing ==================== */

    // TO REMOVE: refactored to Container class
    // if (!fileInput)
    //     postprocessing(npoints, nfeatures, features_float, mean);

    /* free up memory */
    // TO REMOVE: refactored to Container class
    // if (fileInput)
    //     free(features_float[0]);
    // free(features_float);
    // free(features_int[0]);
    // free(features_int);

    // free(cluster_centres_int[0]);
    // free(cluster_centres_int);
    free(cluster_centres[0]);
    free(cluster_centres);
    // free(mean);
    // DPU_ASSERT(dpu_free(allset));

    return output_clusters;
}
