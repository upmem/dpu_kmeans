#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stddef.h>
#include <stdint.h>

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
  // int_feature **clusters_old_int;
  // int_feature **clusters_new_int;
  int64_t *partial_sums_per_dpu;
  int *points_in_clusters_per_dpu;
  bool host_memory_allocated;

 public:
  /**
   * @brief Construct a new Container object
   *
   */
  Container()
      : p(),
        features_float(nullptr),
        features_int(nullptr),
        partial_sums_per_dpu(nullptr),
        points_in_clusters_per_dpu(nullptr),
        host_memory_allocated(false) {
    p.isOutput = 1;
  }

  /**
   * @brief Allocates all DPUs.
   */
  void allocate() { ::allocate(&p); }

  size_t get_ndpu() { return p.ndpu; }
  size_t get_nclusters_round() { return p.nclusters_round; }

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
    transfer_data(verbose);
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
  void load_array_data(py::array_t<int_feature> data_int, uint64_t npoints,
                       int nfeatures, int verbose) {
    int_feature *data_int_ptr = (int_feature *)data_int.request().ptr;

    p.from_file = false;

    p.npoints = npoints;
    p.nfeatures = nfeatures;
    p.npadded = ((p.npoints + 8 * p.ndpu - 1) / (8 * p.ndpu)) * 8 * p.ndpu;

    format_array_input_int(&p, data_int_ptr, &features_int);
    transfer_data(verbose);
  }

  void load_nclusters(unsigned int nclusters) {
    // int_feature *centers_old_int_ptr = (int_feature
    // *)centers_old_int.request().ptr; int_feature *centers_new_int_ptr =
    // (int_feature *)centers_new_int.request().ptr;

    p.nclusters = nclusters;

    // build_jagged_array_int(nclusters, p.nfeatures, centers_old_int_ptr,
    // &clusters_old_int); build_jagged_array_int(nclusters, p.nfeatures,
    // centers_new_int_ptr, &clusters_new_int);
    broadcastNumberOfClusters(&p, nclusters);
    allocateHostMemory();
  }

  void allocateHostMemory() {
    if (host_memory_allocated) deallocateHostMemory();

    partial_sums_per_dpu = (int64_t *)malloc(
        p.nclusters * p.ndpu * p.nfeatures * sizeof(*partial_sums_per_dpu));
    points_in_clusters_per_dpu = (int *)malloc(
        p.ndpu * p.nclusters_round * sizeof(*points_in_clusters_per_dpu));

    host_memory_allocated = true;
  }

  void deallocateHostMemory() {
    free(partial_sums_per_dpu);
    free(points_in_clusters_per_dpu);

    host_memory_allocated = false;
  }

  /**
   * @brief Preprocesses and transfers quantized data to the DPUs
   *
   */
  void transfer_data(int verbose) {
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
  void free_data() {
    /* We are NOT freeing the underlying arrays if they are managed by python
     */
    if (p.from_file) {
      free(features_float[0]);
      free(features_int[0]);
    }
    free(features_float);
    free(features_int);
    // free(p.mean);
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

  void lloyd_iter(py::array_t<int_feature> centers_old_int,
                  py::array_t<int64_t> centers_new_int,
                  py::array_t<int> points_in_clusters) {
    int_feature *old_centers = (int_feature *)centers_old_int.request().ptr;
    int64_t *new_centers = (int64_t *)centers_new_int.request().ptr;
    int *new_centers_len = (int *)points_in_clusters.request().ptr;

    lloydIter(&p, old_centers, new_centers, new_centers_len,
              points_in_clusters_per_dpu, partial_sums_per_dpu);
  }

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
      .def("get_nclusters_round", &Container::get_nclusters_round)
      .def("set_nr_dpus", &Container::set_ndpu)
      .def("load_kernel", &Container::load_kernel)
      .def("load_array_data", &Container::load_array_data)
      .def("load_n_clusters", &Container::load_nclusters)
      .def("free_data", &Container::free_data)
      .def("free_dpus", &Container::free_dpus)
      .def("kmeans", &Container::kmeans_cpp)
      .def("lloyd_iter", &Container::lloyd_iter)
      .def("allocate_host_memory", &Container::allocateHostMemory)
      .def("deallocate_host_memory", &Container::deallocateHostMemory)
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
