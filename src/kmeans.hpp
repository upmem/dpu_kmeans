/**
 * @file kmeans.hpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for the KMeans project
 * @copyright 2024 UPMEM
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <filesystem>
#include <vector>

extern "C" {
#include <dpu_types.h>

#include "common.h"
}

namespace py = pybind11;

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
  bool allocated;       /**< Whether the DPUs are allocated */
  double time_seconds;  /**< Perf counter */
  double cpu_pim_time;  /**< Time to populate the DPUs */
  double pim_cpu_time;  /**< Time to transfer inertia from the CPU */
};

/**
 * @brief Container class for interfacing with python
 *
 * This class holds data that can be reused
 * during different runs of the k-means algorithm.
 */
class Container {
 private:
  kmeans_params p_{}; /**< Struct containing various algorithm parameters. */
  std::vector<int64_t> inertia_per_dpu_; /**< Iteration buffer to read inertia
                                   from the DPUs. */
  std::vector<int> nreal_points_; /* number of real data points on each dpu */

  /**
   * @brief Broadcast current number of clusters to the DPUs
   *
   * @param p Algorithm parameters.
   * @param nclusters Number of clusters.
   */
  void broadcast_number_of_clusters() const;

  /**
   * @brief Fills the DPUs with their assigned points.
   *
   * @param py_features Array: [npoints][nfeatures]
   */
  void populate_dpus(const py::array_t<int_feature> &py_features);

  /**
   * @brief Computes the appropriate task size for DPU tasklets.
   *
   * @param p Algorithm parameters.
   * @return The task size in bytes.
   */
  [[nodiscard]] constexpr auto get_task_size() const -> int;

  /**
   * @brief Broadcasts iteration parameters to the DPUs.
   *
   * @param p Algorithm parameters.
   */
  void broadcast_parameters();

  /**
   * @brief Preprocesses and transfers quantized data to the DPUs.
   */
  void transfer_data(const py::array_t<int_feature> &data_int);

 public:
  Container() = default;

  ~Container() { free_dpus(); }

  Container(const Container &) = delete;
  auto operator=(const Container &) -> Container & = delete;
  Container(Container &&) = default;
  auto operator=(Container &&) -> Container & = default;

  /**
   * @brief Allocates DPUs.
   */
  void allocate();

  [[nodiscard]] auto get_ndpu() const -> size_t { return p_.ndpu; }

  void set_ndpu(uint32_t ndpu) { p_.ndpu = ndpu; }

  void reset_timer() { p_.time_seconds = 0.0; }

  [[nodiscard]] auto get_dpu_run_time() const -> double {
    return p_.time_seconds;
  }
  [[nodiscard]] auto get_cpu_pim_time() const -> double {
    return p_.cpu_pim_time;
  }
  [[nodiscard]] auto get_pim_cpu_time() const -> double {
    return p_.pim_cpu_time;
  }

  /**
   * @brief Loads binary into the DPUs
   *
   * @param binary_path Path to the binary.
   */
  void load_kernel(const std::filesystem::path &binary_path);

  /**
   * @brief Loads data into the DPUs from a python array
   *
   * @param data A python ndarray.
   * @param npoints Number of points.
   * @param nfeatures Number of features.
   * @param threshold Parameter to declare convergence.
   */
  void load_array_data(const py::array_t<int_feature> &data_int,
                       int64_t npoints, int nfeatures);

  /**
   * @brief Informs the DPUs of the number of clusters for that iteration.
   *
   * @param nclusters Number of clusters.
   */
  void load_nclusters(int nclusters);

  /**
   * @brief Frees the DPUs
   */
  void free_dpus();

  /**
   * @brief Runs one iteration of the K-Means Lloyd algorithm.
   *
   * @param old_centers [in] Discretized coordinates of the current
   * centroids.
   * @param centers_psum [out] Sum of points coordinates per cluster per dpu
   * @param centers_pcount [out] Count of elements in each cluster per dpu.
   */
  void lloyd_iter(const py::array_t<int_feature> &old_centers,
                  py::array_t<int64_t> &centers_psum,
                  py::array_t<int> &centers_pcount);

  /**
   * @brief Runs one E step of the K-Means algorithm and gets inertia.
   *
   * @param old_centers [in] Discretized coordinates of the current
   * centroids.
   * @return int64_t The inertia.
   */
  auto compute_inertia(const py::array_t<int_feature> &old_centers) -> int64_t;
};
