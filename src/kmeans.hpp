/**
 * @file kmeans.hpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for the KMeans project
 * @copyright 2024 UPMEM
 */

#pragma once

#include <cstdint>

extern "C" {
#include <dpu_types.h>
}

/**
 * @brief Struct holding various algorithm parameters.
 *
 */
struct kmeans_params {
  int64_t npoints;      /**< Number of points */
  int64_t npadded;      /**< Number of points with padding */
  int64_t npointperdpu; /**< Number of points per dpu */
  int nfeatures;        /**< Number of features */
  int nclusters;        /**< Number of clusters */
  int isOutput;         /**< Whether to print debug information */
  uint32_t ndpu;        /**< Number of allocated dpu */
  dpu_set_t allset;     /**< Struct of the allocated dpu set */
  double time_seconds;  /**< Perf counter */
  double cpu_pim_time;  /**< Time to populate the DPUs */
  double pim_cpu_time;  /**< Time to transfer inertia from the CPU */
};
