#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <iostream>

extern "C" {
#include <dpu.h>

#include "kmeans.h"
}

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

extern "C" int checksum(char *);

namespace py = pybind11;

/**
 * @brief Container class for interfacing with python
 *
 * This class holds data that can be reused
 * during different runs of the k-means algorithm.
 */
class Container {
 private:
  Params p;
  float **features_float;
  int_feature **features_int;

 public:
  /**
   * @brief Construct a new Container object
   *
   */
  Container() : p(), features_float(nullptr), features_int(nullptr) {}

  /**
   * @brief Allocates all DPUs.
   */
  void allocate() { ::allocate(&p); }

  uint32_t get_ndpu() { return p.ndpu; }

  void set_ndpu(uint32_t ndpu) { p.ndpu = ndpu; }

  /**
   * @brief Loads binary into the DPUs
   *
   * @param DPU_BINARY Path to the binary.
   */
  void load_kernel(const char *DPU_BINARY) { ::load_kernel(&p, DPU_BINARY); }
  /**
   * @brief Loads data into the DPU from a file.
   *
   * @param filename Path to the data file.
   * @param is_binary_file Whether the data is encoded as binary.
   */
  void load_file_data(const char *filename, bool is_binary_file,
                      float threshold_in, int verbose) {
    p.from_file = true;
    if (is_binary_file)
      read_bin_input(&p, filename, &features_float);
    else {
      read_txt_input(&p, filename, &features_float);
      save_dat_file(&p, filename, features_float);
    }
    transfer_data(threshold_in, verbose);
    preprocessing(&p, features_float, &features_int, verbose);
  }
  /**
   * @brief Loads data into the DPUs from a python array
   *
   * @param data A python ndarray.
   * @param npoints Number of points.
   * @param nfeatures Number of features.
   * @param threshold Parameter to declare convergence.
   * @param verbose Verbosity level.
   */
  void load_array_data(py::array_t<float> data,
                       py::array_t<int_feature> data_int, uint64_t npoints,
                       int nfeatures, float threshold, float scale_factor,
                       int verbose) {
    float *data_ptr = (float *)data.request().ptr;
    int_feature *data_int_ptr = (int_feature *)data_int.request().ptr;

    p.from_file = false;

    p.npoints = npoints;
    p.nfeatures = nfeatures;
    p.scale_factor = scale_factor;
    format_array_input(&p, data_ptr, &features_float);
    format_array_input_int(&p, data_int_ptr, &features_int);
    transfer_data(threshold, verbose);
  }
  /**
   * @brief Preprocesses and transfers quantized data to the DPUs
   *
   */
  void transfer_data(float threshold, int verbose) {
    p.threshold = threshold;
    p.npointperdpu = p.npadded / p.ndpu;
    populateDpu(&p, features_int);
    broadcastParameters(&p);
    allocateMemory(&p);
#ifdef FLT_REDUCE
    allocateMembershipTable(&p);
#endif
  }
  /**
   * @brief Frees the data.
   */
  void free_data(bool from_file, bool restore_features) {
    /* We are NOT freeing the underlying arrays if they are managed by python
     */
    if (from_file) {
      free(features_float[0]);
      free(features_int[0]);
    } else if (restore_features)
      postprocessing(&p, features_float);
    free(features_float);
    free(features_int);
    free(p.mean);
#ifdef FLT_REDUCE
    deallocateMembershipTable();
#endif
  }
  /**
   * @brief Frees the DPUs
   */
  void free_dpus() {
    ::free_dpus(&p);
    deallocateMemory();
  }

  double get_dpu_run_time() { return p.time_seconds; }

  py::array_t<float> kmeans_cpp(int max_nclusters, int min_nclusters,
                                int isRMSE, int isOutput, int nloops,
                                int max_iter, py::array_t<int> log_iterations,
                                py::array_t<double> log_time);
};

/**
 * @brief Main function for the KMeans algorithm.
 *
 * @return py::array_t<float> The centroids coordinates found by the algorithm.
 */
py::array_t<float> Container ::kmeans_cpp(
    int max_nclusters, /**< upper bound of the number of clusters */
    int min_nclusters, /**< lower bound of the number of clusters */
    int isRMSE,        /**< whether or not RMSE is computed */
    int isOutput,      /**< whether or not to print the centroids */
    int nloops,   /**< how many times the algorithm will be executed for each
                     number of clusters */
    int max_iter, /**< upper bound of the number of iterations */
    py::array_t<int>
        log_iterations, /**< array logging the iterations per nclusters */
    py::array_t<double>
        log_time) /**< array logging the time taken per nclusters */
{
  int best_nclusters = max_nclusters;

  int *log_iter_ptr = (int *)log_iterations.request().ptr;
  double *log_time_ptr = (double *)log_time.request().ptr;

  p.max_nclusters = max_nclusters;
  p.min_nclusters = min_nclusters;
  p.isRMSE = isRMSE;
  p.isOutput = isOutput;
  p.nloops = nloops;
  p.max_iter = max_iter;

  float *clusters = kmeans_c(&p, features_float, features_int, log_iter_ptr,
                             log_time_ptr, &best_nclusters);

  std::vector<ssize_t> shape = {best_nclusters, p.nfeatures};
  std::vector<ssize_t> strides = {(int)sizeof(float) * p.nfeatures,
                                  sizeof(float)};

  py::capsule free_when_done(
      clusters, [](void *f) { delete reinterpret_cast<float *>(f); });

  return py::array_t<float>(shape, strides, clusters, free_when_done);
}

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
      .def("free_data", &Container::free_data)
      .def("free_dpus", &Container::free_dpus)
      .def("kmeans", &Container::kmeans_cpp)
      .def("dpu_run_time", &Container::get_dpu_run_time);

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
