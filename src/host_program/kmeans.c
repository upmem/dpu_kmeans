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

#include "../kmeans.h"

/**
 * @brief Allocates all DPUs
 *
 * @param p Algorithm parameters.
 */
void allocate(Params *p)
{
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &p->allset));
    DPU_ASSERT(dpu_get_nr_dpus(p->allset, &p->ndpu));
}

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
    Params *p,             /**< Algorithm parameters */
    const char *filename,  /**< [in] The file name. */
    float ***features_out) /**< [out] Vector of features. */
{
    float **features;

    FILE *infile;
    if ((infile = fopen(filename, "rb")) == NULL)
    {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
    }

    /* get nfeatures and npoints */
    fread(&p->npoints, sizeof(uint64_t), 1, infile);
    fread(&p->nfeatures, sizeof(int), 1, infile);

    /* rounding the size of the input to the smallest multiple of 8*ndpu larger than npoints */
    p->npadded = ((p->npoints + 8 * p->ndpu - 1) / (8 * p->ndpu)) * 8 * p->ndpu;

    /* allocate space for features[][] and read attributes of all objects */
    features = (float **)malloc(p->npadded * sizeof(*features));
    features[0] = (float *)malloc(p->npadded * p->nfeatures * sizeof(**features));
    for (int ipoint = 1; ipoint < p->npadded; ipoint++)
        features[ipoint] = features[ipoint - 1] + p->nfeatures;

    /* checking that we managed to assign enough memory */
    if (!features[0])
    {
        perror("malloc features[0]");
        exit(EXIT_FAILURE);
    }

    fread(features[0], sizeof(float), p->npoints * p->nfeatures, infile);

    fclose(infile);

    *features_out = features;
}

/**
 * @brief Reads a text input file from disk.
 */
void read_txt_input(
    Params *p,             /**< Algorithm parameters */
    const char *filename,  /**< [in] The file name. */
    float ***features_out) /**< [out] Vector of features. */
{
    char line[1024];
    float **features;

    FILE *infile;
    if ((infile = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
        if (strtok(line, " \t\n") != 0)
            p->npoints++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL)
    {
        if (strtok(line, " \t\n") != 0)
        {
            /* ignore the id (first attribute): nfeatures = 1; */
            while (strtok(NULL, " ,\t\n") != NULL)
                p->nfeatures++;
            break;
        }
    }
    /* rounding the size of the input to the smallest multiple of 8*ndpu larger than npoints */
    p->npadded = ((p->npoints + 8 * p->ndpu - 1) / (8 * p->ndpu)) * 8 * p->ndpu;

    /* allocate space for features[] and read attributes of all objects */
    features = (float **)malloc(p->npadded * sizeof(*features));
    features[0] = (float *)malloc(p->npadded * p->nfeatures * sizeof(**features));
    for (int ipoint = 1; ipoint < p->npadded; ipoint++)
        features[ipoint] = features[ipoint - 1] + p->nfeatures;

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
            if (strtok(line, " \t\n") == NULL)
                continue;
            for (int ifeature = 0; ifeature < p->nfeatures; ifeature++)
            {
                features[0][ifeature_global] = atof(strtok(NULL, " ,\t\n"));
                ifeature_global++;
            }
        }
    }
    fclose(infile);

    *features_out = features;
}

/**
 * @brief Saves the input data in a binary file for faster access next time.
 *
 * @param p Algorithm parameters.
 * @param filename_in [in] Name of the input text file.
 * @param features [npoints][nfeatures] Feature array.
 */
void save_dat_file(Params *p, const char *filename_in, float **features)
{
    char *filename = strdup(filename_in);
    char suffix[] = ".dat";

    int n = strlen(filename) + strlen(suffix);
    char *dat_name = (char *)malloc(n * sizeof(*dat_name));

    strcpy(dat_name, filename);
    strip_ext(dat_name);
    strcat(dat_name, ".dat");

    printf("Writing points in binary format to %s\n", dat_name);

    FILE *binfile;
    binfile = fopen(dat_name, "wb");
    fwrite(&p->npoints, sizeof(p->npoints), 1, binfile);
    fwrite(&p->nfeatures, sizeof(p->nfeatures), 1, binfile);
    fwrite(features[0], sizeof(*features[0]), p->npoints * p->nfeatures, binfile);
    fclose(binfile);

    free(filename);
    free(dat_name);
}

/**
 * @brief Formats a flat array into a bidimensional representation
 */
void format_array_input(
    Params *p,             /**< Algorithm parameters. */
    float *data,           /**< [in] The data as a flat table */
    float ***features_out) /**< [out] The data as two dimensional table */
{
    // uint64_t npadded;
    p->npadded = ((p->npoints + 8 * p->ndpu - 1) / (8 * p->ndpu)) * 8 * p->ndpu;

    float **features = (float **)malloc(p->npadded * sizeof(*features));
    features[0] = data;
    for (int ipoint = 1; ipoint < p->npadded; ipoint++)
        features[ipoint] = features[ipoint - 1] + p->nfeatures;

    *features_out = features;
}

/**
 * @brief Preprocesses the data before running the KMeans algorithm.
 *
 * @return float Scaling factor applied to the input data.
 */
void preprocessing(
    Params *p,                       /**< Algorithm parameters */
    float **features_float,          /**< [in] Features as floats. */
    int_feature ***features_int_out, /**< [out] Features as integers. */
    int verbose)                     /**< [in] Whether or not to print runtime information. */
{
    uint64_t ipoint;
    int ifeature;

    float *mean;
    float *variance;
    int_feature **features_int;
    float avg_variance;
    float max_feature = 0;

#ifdef PERF_COUNTER
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
#endif

    p->npointperdpu = p->npadded / p->ndpu;

    /* DEBUG : print features head */
    // printf("features head:\n");
    // for (int ipoint = 0; ipoint < 10; ipoint++)
    // {
    //     for (int ifeature = 0; ifeature < nfeatures; ifeature++)
    //         printf("%.4f ", features[ipoint][ifeature]);
    //     printf("\n");
    // }
    // printf("\n");

    mean = (float *)calloc(p->nfeatures, sizeof(*p->mean));
    variance = (float *)calloc(p->nfeatures, sizeof(*variance));
/* compute mean by feature */
#pragma omp parallel for collapse(2) \
    reduction(+                      \
              : mean[:p->nfeatures])
    for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
        for (ipoint = 0; ipoint < p->npoints; ipoint++)
            mean[ifeature] += features_float[ipoint][ifeature];

#pragma omp parallel for
    for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
        mean[ifeature] /= p->npoints;

    p->mean = mean;

    if (verbose)
    {
        printf("means = ");
        for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
            printf(" %.4f", p->mean[ifeature]);
        printf("\n");
    }

    /* subtract mean from each feature */
#pragma omp parallel for collapse(2)
    for (ipoint = 0; ipoint < p->npoints; ipoint++)
        for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
            features_float[ipoint][ifeature] -= p->mean[ifeature];

    /* ****** discretization ****** */

    /* get maximum absolute value of features */
#pragma omp parallel for collapse(2) \
    reduction(max                    \
              : max_feature)
    for (ipoint = 0; ipoint < p->npoints; ipoint++)
        for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
            if (fabsf(features_float[ipoint][ifeature]) > max_feature)
                max_feature = fabsf(features_float[ipoint][ifeature]);
    switch (sizeof(int_feature))
    {
    case 1UL:
        p->scale_factor = INT8_MAX / max_feature / 2;
        break;
    case 2UL:
        p->scale_factor = INT16_MAX / max_feature / 2;
        break;
    case 4UL:
        p->scale_factor = INT32_MAX / max_feature / 2;
        break;
    default:
        printf("Error: unsupported type for int_feature.\n");
        exit(0);
    }

    if (verbose)
    {
        printf("max absolute value : %f\n", max_feature);
        printf("scale factor = %.4f\n", p->scale_factor);
    }

    /* allocate space for features_int[][] and convert attributes of all objects */
    features_int = (int_feature **)malloc(p->npadded * sizeof(*features_int));
    features_int[0] = (int_feature *)malloc(p->npadded * p->nfeatures * sizeof(features_int));
    for (ipoint = 1; ipoint < p->npadded; ipoint++)
        features_int[ipoint] = features_int[ipoint - 1] + p->nfeatures;

    /* checking that we managed to assign enough memory */
    if (!features_int[0])
    {
        perror("malloc features_int[0]");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for collapse(2)
    for (ipoint = 0; ipoint < p->npoints; ipoint++)
        for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
            features_int[ipoint][ifeature] = lroundf(features_float[ipoint][ifeature] * p->scale_factor);

    /* DEBUG : print features head */
    // printf("features head:\n");
    // for (int ipoint = 0; ipoint < (npoints >= 10 ? 10 : npoints); ipoint++)
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
              : variance[:p->nfeatures])
    for (ipoint = 0; ipoint < p->npoints; ipoint++)
        for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
            variance[ifeature] += features_float[ipoint][ifeature] * features_float[ipoint][ifeature];

#pragma omp parallel for
    for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
        variance[ifeature] /= p->npoints;

    /* compute average of variance */
    avg_variance = 0;
#pragma omp parallel for reduction(+ \
                                   : avg_variance)
    for (ifeature = 0; ifeature < p->nfeatures; ifeature++)
        avg_variance += variance[ifeature];
    avg_variance /= p->nfeatures;
    p->threshold *= avg_variance;

#ifdef PERF_COUNTER
    /* compute time spent on preprocessing */
    gettimeofday(&toc, NULL);
    printf("preprocessing time: %f seconds\n\n", time_seconds(tic, toc));
#endif

    if (verbose)
    {
        printf("avg_variance = %.4f\n", avg_variance);
        printf("threshold = %.4f\n", p->threshold);
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

    *features_int_out = features_int;
}

/**
 * @brief Restores the input data to its original state.
 *
 * @param p [in] Algorithm parameters.
 * @param features [in,out] array of features
 */
void postprocessing(Params *p, float **features_float)
{
#pragma omp parallel for collapse(2)
    for (uint64_t ipoint = 0; ipoint < p->npoints; ipoint++)
        for (int ifeature = 0; ifeature < p->nfeatures; ifeature++)
            features_float[ipoint][ifeature] += p->mean[ifeature];
}

/**
 * @brief Checks for errors in the input
 *
 * @param p Algorithm parameters.
 */
static void error_check(Params *p)
{
    if (p->npoints < p->min_nclusters)
    {
        printf("Error: min_nclusters(%d) > npoints(%lu) -- cannot proceed\n", p->min_nclusters, p->npoints);
        exit(EXIT_FAILURE);
    }
    if ((p->max_nclusters < p->min_nclusters) || (p->max_nclusters > ASSUMED_NR_CLUSTERS))
    {
        printf("Error: min_nclusters(%d) > max_nclusters(%lu) or max_nclusters > max clusters allowed(%d) -- cannot proceed\n", p->min_nclusters, p->npoints, ASSUMED_NR_CLUSTERS);
        exit(EXIT_FAILURE);
    }
    if (ASSUMED_NR_FEATURES < p->nfeatures)
    {
        printf("Error: nfeatures(%d) > max clusters allowed(%d) -- cannot proceed\n", p->nfeatures, ASSUMED_NR_FEATURES);
        exit(EXIT_FAILURE);
    }
    if (p->npadded * p->nfeatures / p->ndpu > MAX_FEATURE_DPU)
    {
        printf("Error: npadded*nfeatures/ndpu(%lu) > max features allowed per dpu(%d) -- cannot proceed\n", p->npadded * p->nfeatures / p->ndpu, MAX_FEATURE_DPU);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Output to array.
 *
 * @param p Algorithm parameters.
 * @param best_nclusters [in] Best number of clusters according to RMSE.
 * @param cluster_centres [in] Coordinate of clusters centres for the best iteration.
 * @return float* The return array
 */
static float *array_output(Params *p, int best_nclusters, float **cluster_centres)
{
#pragma omp parallel for collapse(2)
    for (int icluster = 0; icluster < best_nclusters; icluster++)
        for (int ifeature = 0; ifeature < p->nfeatures; ifeature++)
            cluster_centres[icluster][ifeature] = cluster_centres[icluster][ifeature] + p->mean[ifeature];

    return cluster_centres[0];
}

/**
 * @brief Output to the command line.
 *
 */
static void cli_output(
    Params *p,               /**< Algorithm parameters */
    float **cluster_centres, /**< [in] coordinate of clusters centres for the best iteration */
    float rmse,              /**< [in] value of the RMSE for the best iteration */
    int index)               /**< [in] number of trials for the best RMSE */
{
    /* print cluster center coordinates */
    if (p->min_nclusters == p->max_nclusters)
    {
        printf("\n================= Centroid Coordinates =================\n");
        for (int icluster = 0; icluster < p->max_nclusters; icluster++)
        {
            printf("%2d:", icluster);
            for (int ifeature = 0; ifeature < p->nfeatures; ifeature++)
                printf(" % 10.6f", cluster_centres[icluster][ifeature]);
            printf("\n");
        }
    }

    printf("Number of Iteration: %d\n", p->nloops);

    if (p->min_nclusters == p->max_nclusters && p->isRMSE)
    {
        if (p->nloops != 1)
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
    Params *p,                  /**< Algorithm parameters */
    float **features_float,     /**< [in] array of features  */
    int_feature **features_int, /**< [in] array of quantized features */
    int *log_iterations,        /**< [out] Number of iterations per nclusters */
    double *log_time,           /**< [out] Time taken per nclusters */
    int *best_nclusters)        /**< [out] best number of clusters according to RMSE */
{
    /* Variables for I/O. */
    float *output_clusters; /* return pointer */

    /* Data arrays. */
    float **cluster_centres = NULL; /* array of centroid coordinates */

    /* Generated values. */
    int index;  /* number of iterations on the best run */
    float rmse; /* RMSE value */

    if (p->isOutput)
    {
        printf("\nNumber of objects without padding: %lu\n", p->npoints);
        printf("Number of objects with padding: %lu\n", p->npadded);
        printf("Number of features: %d\n", p->nfeatures);
        printf("Number of DPUs: %d\n", p->ndpu);
    }

    /* Error check for clusters. */
    error_check(p);

    /* ======================= core of the clustering ===================*/

    cluster_centres = NULL;
    index = cluster(
        p,                /* Algorithm parameters */
        features_float,   /* [in] array: [npoints][nfeatures] */
        features_int,     /* [in] array: [npoints][nfeatures] */
        best_nclusters,   /* [out] number between min and max */
        &cluster_centres, /* [out] [best_nclusters][nfeatures] */
        &rmse,            /* [out] Root Mean Squared Error */
        log_iterations,   /* [out] log of the number of iterations */
        log_time          /* [out] log of the time taken */
    );

    /* =============== Array Output ====================== */

    output_clusters = array_output(p, *best_nclusters, cluster_centres);

    /* =============== Command Line Output =============== */

    if (p->isOutput)
        cli_output(p, cluster_centres, rmse, index);

    free(cluster_centres);

    return output_clusters;
}
