/**
 * @file kmeans.hpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for the KMeans project
 * @copyright 2024 UPMEM
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstdint>
#include <filesystem>
#include <optional>
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
  double time_seconds;  /**< Perf counter */
  double cpu_pim_time;  /**< Time to populate the DPUs */
  double pim_cpu_time;  /**< Time to transfer inertia from the CPU */
};

namespace fs = std::filesystem;

/**
 * @brief Container class for interfacing with python
 *
 * This class holds data that can be reused
 * during different runs of the k-means algorithm.
 */
class Container {
 private:
  kmeans_params p_{};                    /**< Algorithm parameters. */
  std::optional<dpu_set_t> allset_{};    /**< Set of DPUs. */
  uint32_t requested_dpus_{0};           /**< Number of requested DPUs. */
  std::vector<int64_t> inertia_per_dpu_; /**< Internal iteration buffer. */
  std::vector<int> nreal_points_;        /**< Real data points per dpu. */
  std::optional<std::string> hash_;      /**< Hash of the data. */
  std::optional<fs::path> binary_path_;  /**< Path to the binary. */
  std::optional<size_t> data_size_;      /**< Size of the data. */

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
   * @brief Allocates DPUs if necessary
   *
   * @param ndpu Number of DPUs to allocate. 0 means all available DPUs.
   */
  void allocate(uint32_t ndpu);

  [[nodiscard]] auto get_ndpu() const -> size_t { return p_.ndpu; }

  [[nodiscard]] auto allocated() const -> bool { return allset_.has_value(); }

  [[nodiscard]] auto hash() const -> std::optional<py::bytes> { return hash_; }

  [[nodiscard]] auto binary_path() const -> std::optional<fs::path> {
    return binary_path_;
  }

  [[nodiscard]] auto data_size() const -> std::optional<size_t> {
    return data_size_;
  }

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
                       const std::string &hash);

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
   * @brief Get the inertia computed in the E step of the previous iteration.
   *
   * @return int64_t The inertia.
   */
  auto get_inertia() -> int64_t;
};
