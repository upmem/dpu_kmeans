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
#include <pybind11/stl/filesystem.h>

#include <filesystem>

#include "host_program/dimm_manager.hpp"
#include "host_program/lloyd_iter.hpp"
#include "kmeans.hpp"

extern "C" {
#include <dpu.h>

#include "common.h"
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
  std::vector<int64_t> inertia_per_dpu_; /**< Iteration buffer to read inertia
                                   from the DPUs. */

  /**
   * @brief Preprocesses and transfers quantized data to the DPUs.
   */
  void transfer_data(const py::array_t<int_feature> &data_int) {
    populate_dpus(p_, data_int);
    broadcast_parameters(p_);
  }

 public:
  Container() = default;

  /**
   * @brief Allocates all DPUs.
   */
  void allocate() {
    ::allocate_dpus(p_);
    inertia_per_dpu_.resize(p_.ndpu);
  }

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
  void load_kernel(const std::filesystem::path &binary_path) {
    load_kernel_internal(p_, binary_path);
  }

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

    broadcast_number_of_clusters(p_, nclusters);
  }

  /**
   * @brief Frees the DPUs
   */
  void free_dpus() { ::free_dpus(p_); }

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
                  py::array_t<int> &centers_pcount) {
    ::lloyd_iter(p_, old_centers, centers_psum, centers_pcount);
  }

  /**
   * @brief Runs one E step of the K-Means algorithm and gets inertia.
   *
   * @param old_centers [in] Discretized coordinates of the current
   * centroids.
   * @return int64_t The inertia.
   */
  auto compute_inertia(const py::array_t<int_feature> &old_centers) -> int64_t {
    return lloyd_iter_with_inertia(p_, old_centers, inertia_per_dpu_);
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
      .def_property("nr_dpus", &Container::get_ndpu, &Container::set_ndpu)
      .def("load_kernel", &Container::load_kernel)
      .def("load_array_data", &Container::load_array_data)
      .def("load_n_clusters", &Container::load_nclusters)
      .def("free_dpus", &Container::free_dpus)
      .def("lloyd_iter", &Container::lloyd_iter)
      .def("compute_inertia", &Container::compute_inertia)
      .def("reset_timer", &Container::reset_timer)
      .def_property_readonly("dpu_run_time", &Container::get_dpu_run_time)
      .def_property_readonly("cpu_pim_time", &Container::get_cpu_pim_time)
      .def_property_readonly("pim_cpu_time", &Container::get_pim_cpu_time);

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
