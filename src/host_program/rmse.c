/**
 * @file rmse.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Computes the RMSE of a clustering.
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "../kmeans.h"

/**
 * @brief multi-dimensional spatial Euclid distance square
 *
 * @return the squared distance
 */
__inline float euclid_dist_2(float *pt1,  /**< first point */
                             float *pt2,  /**< second point */
                             int numdims) /**< number of dimensions */
{
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++)
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

    return (ans);
}

/**
 * @brief find the point closest to a specific point
 *
 * @return the point index closest to the input point
 */
__inline int find_nearest_point(float *pt,     /**< array:[nfeatures] */
                                int nfeatures, /**< number of features */
                                float **pts,   /**< array:[npts][nfeatures] */
                                int npts)      /**< number of points */
{
    int index, i;
    float max_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++)
    {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures); /* no need square root */
        if (dist < max_dist)
        {
            max_dist = dist;
            index = i;
        }
    }
    return (index);
}

/**
 * @brief calculates RMSE of clustering
 *
 * @return the RMSE
 */
float rms_err(Params *p,
              float **feature,         /**< array:[npoints][nfeatures] */
              float **cluster_centres, /**< array:[nclusters][nfeatures] */
              int nclusters)           /**< number of clusters */
{
    int i;
    int nearest_cluster_index; /* cluster center id with min distance to pt */
    float sum_euclid = 0.0;    /* sum of Euclidean distance squares */
    float ret;                 /* return value */
    uint64_t npoints = p->npoints;
    int nfeatures = p->nfeatures;

/* calculate and sum the square of euclidean distance*/
#pragma omp parallel for shared(feature, cluster_centres)                         \
    firstprivate(npoints, nfeatures, nclusters) private(i, nearest_cluster_index) \
        reduction(+                                                               \
                  : sum_euclid)                                                   \
            schedule(static)
    for (i = 0; i < npoints; i++)
    {
        nearest_cluster_index = find_nearest_point(feature[i],
                                                   nfeatures,
                                                   cluster_centres,
                                                   nclusters);

        sum_euclid += euclid_dist_2(feature[i],
                                    cluster_centres[nearest_cluster_index],
                                    nfeatures);
    }
    /* divide by n, then take sqrt */
    ret = sqrt(sum_euclid / p->npoints);

    return (ret);
}
