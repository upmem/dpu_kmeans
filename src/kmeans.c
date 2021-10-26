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

struct dpu_set_t allset; /**< Set of all available DPUs. */

/**
 * @brief Returns the seconds elapsed between two timeval structures.
 *
 * @param tic First timeval.
 * @param toc Second timeval.
 * @return Elapsed time in seconds.
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
 * @param fname The file name string.
 */
void strip_ext(char *fname)
{
    char *end = fname + strlen(fname);

    while (end > fname && *end != '.')
        --end;

    if (end > fname)
        *end = '\0';
}

/**
 * @brief Usage info on the KMeans program.
 *
 * @param argv0
 */
void usage(char *argv0)
{
    char *help =
        "\nUsage: %s [switches] -i filename\n\n"
        "    -i filename      :file containing data to be clustered\n"
        "    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
        "    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
        "    -t threshold     :threshold value                       [default=0.001]\n"
        "    -l nloops        :iteration for each number of clusters [default=1]\n"
        "    -b               :input file is in binary format         [default=0]\n"
        "    -r               :calculate RMSE                        [default=off]\n"
        "    -o               :output cluster center coordinates     [default=off]\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/**
 * @brief Reads a binary input file from disk.
 *
 * @param filename The file name.
 * @param npoints_out [out] Number of points.
 * @param npadded_out [out] Number of points with padding added.
 * @param nfeatures_out [out] Number of features.
 * @param ndpu Number of available DPUs.
 * @param features_out [out] Vector of features.
 */
void read_binary_input(
    char *filename,
    uint64_t *npoints_out,
    uint64_t *npadded_out,
    int *nfeatures_out,
    uint32_t ndpu,
    float ***features_out)
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
 *
 * @param filename The file name.
 * @param npoints_out [out] Number of points.
 * @param npadded_out [out] Number of points with padding added.
 * @param nfeatures_out [out] Number of features.
 * @param ndpu Number of available DPUs.
 * @param features_out [out] Vector of features.
 */
void read_text_input(
    char *filename,
    uint64_t *npoints_out,
    uint64_t *npadded_out,
    int *nfeatures_out,
    uint32_t ndpu,
    float ***features_out)
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
 * @brief Preprocesses the data before running the KMeans algorithm.
 *
 * @param mean_out [out] Per-feature average.
 * @param nfeatures Number of features.
 * @param npoints Number of points.
 * @param npadded Number of points with padding.
 * @param features Features as floats.
 * @param features_int_out [out] Features as integers.
 * @param threshold [out] Termination criterion for the algorithm.
 * @return Scaling factor applied to the input data.
 */
float preprocessing(
    float **mean_out,
    int nfeatures,
    uint64_t npoints,
    uint64_t npadded,
    float **features,
    int_feature ***features_int_out,
    float *threshold)
{
    int ipoint, ifeature;

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
            reduction(+ : mean[:nfeatures])
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
            reduction(max : max_feature)
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

    printf("max absolute value : %f\n", max_feature);
    printf("scale factor = %.4f\n", scale_factor);

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
            reduction(+ : variance[:nfeatures])
    for (ipoint = 0; ipoint < npoints; ipoint++)
        for (ifeature = 0; ifeature < nfeatures; ifeature++)
            variance[ifeature] += (double)features_int[ipoint][ifeature] * features_int[ipoint][ifeature];

    #pragma omp parallel for
    for (ifeature = 0; ifeature < nfeatures; ifeature++)
        variance[ifeature] /= npoints;

    /* compute average of variance */
    avg_variance = 0;
    #pragma omp parallel for reduction(+ : avg_variance)
    for (ifeature = 0; ifeature < nfeatures; ifeature++)
        avg_variance += variance[ifeature];
    avg_variance /= nfeatures;
    *threshold *= avg_variance;

    #ifdef PERF_COUNTER
    /* compute time spent on preprocessing */
    gettimeofday(&toc, NULL);
    printf("preprocessing time: %f seconds\n\n", time_seconds(tic, toc));
    #endif

    printf("avg_variance = %.4f\n", avg_variance);
    printf("threshold = %.4f\n", *threshold);
    printf("\npreprocessing completed\n\n");

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
 * @brief Main function for the KMeans algorithm.
 *
 * Execute ./kmeans for usage information.
 * @param argc n° of command line arguments
 * @param argv command line arguments
 * @return int 0 on normal exit
 */
int kmeans_c(int argc, char **argv)
{
    printf("helloword\n");

    /* Variables for I/O. */
    int opt;
    extern char *optarg;
    char *filename = 0;
    int isBinaryFile = 0;
    char testname[100];
    int isOutput = 0;

    /* Default parameters. */
    float threshold = 0.0001; /* threshold for termination of the algorithm */
    int max_nclusters = 5;      /* upper bound of the number of clusters */
    int min_nclusters = 5;      /* lower bound of the number of clusters */
    int best_nclusters = 0;      /* best number of clusters according to RMSE */
    int nloops = 1;              /* how many times the algorithm will be executed for each number of clusters */
    int isRMSE = 0;              /* whether or not RMSE is computed */

    /* Size variables. */
    int nfeatures;      /* number of features */
    uint64_t npoints; /* number of points */
    uint64_t npadded; /* number of points with padding */
    uint32_t ndpu;      /* number of available DPUs */

    /* Data arrays. */
    float **features;                          /* array of features */
    int_feature **features_int;                  /* array of discretized features */
    float **cluster_centres = NULL;              /* array of centroid coordinates */
    int_feature **cluster_centres_int = NULL; /* array of discretized centroid coordinates */
    float *mean;                              /* feature-wise average of points coordinates */

    int index;    /* number of iterations on the best run */
    float rmse; /* RMSE value */

    float scale_factor; /* scaling factor of the input features */

    /* obtain command line arguments and change appropriate options */
    while ((opt = getopt(argc, argv, "i:t:m:n:l:bro")) != EOF)
    {
        switch (opt)
        {
        case 'i':
            filename = optarg;
            break;
        case 'b':
            isBinaryFile = 1;
            break;
        case 't':
            threshold = atof(optarg);
            break;
        case 'm':
            max_nclusters = atoi(optarg);
            break;
        case 'n':
            min_nclusters = atoi(optarg);
            break;
        case 'r':
            isRMSE = 1;
            break;
        case 'o':
            isOutput = 1;
            break;
        case 'l':
            nloops = atoi(optarg);
            break;
        case '?':
            usage(argv[0]);
            break;
        default:
            usage(argv[0]);
            break;
        }
    }

    if (filename == 0)
        usage(argv[0]);

    /* ============== DPUs init ==============*/
    /* necessary to do it first to know the n° of available DPUs */
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &allset));
    DPU_ASSERT(dpu_get_nr_dpus(allset, &ndpu));
    /* ============== DPUs init end ==========*/

    /* ============== I/O begin ==============*/
    if (isBinaryFile)
    { //Binary file input
        read_binary_input(filename, &npoints, &npadded, &nfeatures, ndpu, &features);
    }
    else
    { //Text file input
        read_text_input(filename, &npoints, &npadded, &nfeatures, ndpu, &features);

        /* Saving features as a binary for next time */
        strip_ext(filename);
        strcat(filename, ".dat");
        FILE *binfile;
        binfile = fopen(filename, "wb");
        fwrite(&npoints, sizeof(uint64_t), 1, binfile);
        fwrite(&nfeatures, sizeof(int), 1, binfile);
        fwrite(features[0], sizeof(float), npoints * nfeatures, binfile);
        fclose(binfile);
    }

    strip_ext(filename);
    strcpy(testname, basename(filename));
    filename = dirname(filename);
    strcat(filename, "/kmeanstime_dpu_");
    strcat(filename, testname);
    strcat(filename, ".log");

    printf("\nI/O completed\n");
    printf("\nNumber of objects without padding: %lu\n", npoints);
    printf("Number of objects with padding: %lu\n", npadded);
    printf("Number of features: %d\n", nfeatures);
    printf("Number of DPUs: %d\n", ndpu);
    /* ============== I/O end ==============*/

    // error check for clusters
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

    /* ======================= pre-processing ===========================*/
    scale_factor = preprocessing(&mean, nfeatures, npoints, npadded, features, &features_int, &threshold);
    /* ======================= pre-processing end =======================*/

    /* ======================= core of the clustering ===================*/

    cluster_centres = NULL;
    index = cluster(npoints,          /* number of data points */
                    npadded,          /* number of data points with padding */
                    nfeatures,          /* number of features for each point */
                    ndpu,              /* number of available DPUs */
                    features,          /* array: [npoints][nfeatures] */
                    features_int,      /* array: [npoints][nfeatures] */
                    min_nclusters,      /* range of min to max number of clusters */
                    max_nclusters,      /* range of min to max number of clusters */
                    threshold,          /* loop termination factor */
                    &best_nclusters,  /* return: number between min and max */
                    &cluster_centres, /* return: [best_nclusters][nfeatures] */
                    &rmse,              /* Root Mean Squared Error */
                    isRMSE,              /* calculate RMSE */
                    nloops,              /* number of iteration for each number of clusters */
                    filename);          /* name of the log file */

    /* =============== Command Line Output =============== */

    /* cluster center coordinates
       :displayed only for when k=1*/
    if ((min_nclusters == max_nclusters) && (isOutput == 1))
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

    /* free up memory */
    free(features[0]);
    free(features);
    free(features_int[0]);
    free(features_int);
    // free(cluster_centres_int[0]);
    // free(cluster_centres_int);
    free(cluster_centres[0]);
    free(cluster_centres);
    free(mean);
    DPU_ASSERT(dpu_free(allset));
    return (0);
}
