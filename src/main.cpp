/**
 * @file main.cpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Binding file for the KMeans project
 * @copyright 2021 UPMEM
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>

#include "host_program/dimm_manager.hpp"

extern "C" {
#include <dpu.h>

#include "kmeans.h"
}

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

[[nodiscard]] constexpr auto add(int i, int j) -> int { return i + j; }

extern "C" auto checksum(char *) -> int;

namespace py = pybind11;

/**
 * @brief Container class for interfacing with python
 *
 * This class holds data that can be reused
 * during different runs of the k-means algorithm.
 */
class Container {
 private:
  kmeans_params p_{}; /**< Struct containing various algorithm parameters. */
  std::vector<int64_t> partial_sums_per_dpu_;   /**< Iteration buffer to read
                                       feature sums from the DPUs. */
  std::vector<int> points_in_clusters_per_dpu_; /**< Iteration buffer to read
                                       cluster counts from the DPUs. */
  std::vector<uint64_t> inertia_per_dpu_; /**< Iteration buffer to read inertia
                                   from the DPUs. */
  bool host_memory_allocated_{}; /**< Whether the iteration buffers have been
                                 allocated. */

  /**
   * @brief Preprocesses and transfers quantized data to the DPUs.
   */
  void transfer_data(const py::array_t<int_feature> &data_int) {
    populate_dpus(&p_, data_int);
    broadcastParameters(&p_);
#ifdef FLT_REDUCE
    allocateMembershipTable(&p);
#endif
  }

 public:
  Container() = default;

  /**
   * @brief Allocates all DPUs.
   */
  void allocate() { ::allocate_dpus(&p_); }

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
  void load_kernel(const char *binary_path) { ::load_kernel(&p_, binary_path); }

  /**
   * @brief Loads data into the DPUs from a python array
   *
   * @param data A python ndarray.
   * @param npoints Number of points.
   * @param nfeatures Number of features.
   * @param threshold Parameter to declare convergence.
   */
  void load_array_data(const py::array_t<int_feature> &data_int,
                       int64_t npoints, int nfeatures) {
    p_.npoints = npoints;
    p_.nfeatures = nfeatures;
    p_.npadded = ((p_.npoints + 8 * p_.ndpu - 1) / (8 * p_.ndpu)) * 8 * p_.ndpu;
    p_.npointperdpu = p_.npadded / p_.ndpu;

    transfer_data(data_int);
  }

  /**
   * @brief Informs the DPUs of the number of clusters for that iteration.
   *
   * @param nclusters Number of clusters.
   */
  void load_nclusters(int nclusters) {
    p_.nclusters = nclusters;

    broadcastNumberOfClusters(&p_, nclusters);
    allocate_host_memory();
  }

  /**
   * @brief Allocates host iteration buffers.
   *
   */
  void allocate_host_memory() {
    if (host_memory_allocated_) {
      deallocate_host_memory();
    }

    /* allocate buffer to read coordinates sums from the DPUs */
    partial_sums_per_dpu_.resize(p_.nclusters * p_.ndpu * p_.nfeatures);

    /* allocate buffer to read clusters counts from the DPUs */
    size_t count_in_8bytes = 8 / sizeof(points_in_clusters_per_dpu_.back());
    size_t nclusters_aligned =
        ((p_.nclusters + count_in_8bytes - 1) / count_in_8bytes) *
        count_in_8bytes;
    points_in_clusters_per_dpu_.resize(p_.ndpu * nclusters_aligned);

    /* allocate buffer to read inertia from the DPUs */
    inertia_per_dpu_.resize(p_.ndpu);

    host_memory_allocated_ = true;
  }

  /**
   * @brief Frees the host iteration buffers.
   *
   */
  void deallocate_host_memory() {
    partial_sums_per_dpu_.clear();
    points_in_clusters_per_dpu_.clear();
    inertia_per_dpu_.clear();

    host_memory_allocated_ = false;
  }

  /**
   * @brief Frees the data.
   * Only the jagged pointers are freed. The feature values themselves are
   * managed by Python.
   */
  void free_data() {
#ifdef FLT_REDUCE
    deallocateMembershipTable();
#endif
  }

  /**
   * @brief Frees the DPUs
   */
  void free_dpus() { ::free_dpus(&p_); }

  /**
   * @brief Runs one iteration of the K-Means Lloyd algorithm.
   *
   * @param centers_old_int [in] Discretized coordinates of the current
   * centroids.
   * @param centers_new_int [out] Discretized coordinates of the updated
   * centroids (before division by number of points).
   * @param points_in_clusters [out] Counts of points per cluster.
   */
  void lloyd_iter(const py::array_t<int_feature> &centers_old_int,
                  const py::array_t<int64_t> &centers_new_int,
                  const py::array_t<int> &points_in_clusters) {
    int_feature *old_centers =
        static_cast<int_feature *>(centers_old_int.request().ptr);
    int64_t *new_centers =
        static_cast<int64_t *>(centers_new_int.request().ptr);
    int *new_centers_len = static_cast<int *>(points_in_clusters.request().ptr);

    lloydIter(&p_, old_centers, new_centers, new_centers_len,
              points_in_clusters_per_dpu_.data(), partial_sums_per_dpu_.data());
  }

  /**
   * @brief Runs one E step of the K-Means algorithm and gets inertia.
   *
   * @param centers_old_int [in] Discretized coordinates of the current
   * centroids.
   * @return uint64_t The inertia.
   */
  auto compute_inertia(const py::array_t<int_feature> &centers_old_int)
      -> uint64_t {
    int_feature *old_centers =
        static_cast<int_feature *>(centers_old_int.request().ptr);
    return lloydIterWithInertia(&p_, old_centers, inertia_per_dpu_.data());
  }
};

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        DPU kmeans plugin
        -----------------

        .. currentmodule:: dpu_kmeans._core

        .. autosummary::
           :toctree: _generate

           add
           subtract
           checksum
           Container
    )pbdoc";

  py::class_<Container>(m, "Container", R"pbdoc(
        Container object to interface with the DPUs
    )pbdoc")
      .def(py::init<>())
      .def("allocate", &Container::allocate)
      .def("get_nr_dpus", &Container::get_ndpu)
      .def("set_nr_dpus", &Container::set_ndpu)
      .def("load_kernel", &Container::load_kernel)
      .def("load_array_data", &Container::load_array_data)
      .def("load_n_clusters", &Container::load_nclusters)
      .def("free_data", &Container::free_data)
      .def("free_dpus", &Container::free_dpus)
      .def("lloyd_iter", &Container::lloyd_iter)
      .def("compute_inertia", &Container::compute_inertia)
      .def("allocate_host_memory", &Container::allocate_host_memory)
      .def("deallocate_host_memory", &Container::deallocate_host_memory)
      .def("reset_timer", &Container::reset_timer)
      .def("get_dpu_run_time", &Container::get_dpu_run_time)
      .def("get_cpu_pim_time", &Container::get_cpu_pim_time)
      .def("get_pim_cpu_time", &Container::get_pim_cpu_time);

  m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

  m.def(
      "subtract", [](int i, int j) { return i - j; },
      R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

  m.def("checksum", &checksum, R"pbdoc(
        Checksum test on dpus
    )pbdoc");

  m.attr("FEATURE_TYPE") = py::int_(FEATURE_TYPE);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
