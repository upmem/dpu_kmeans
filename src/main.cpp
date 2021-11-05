#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

extern "C"
{
#include <dpu.h>
#include "kmeans.h"
}

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j)
{
    return i + j;
}

int array_sum()
{
    return 0;
}

extern "C" char *call_home(char *);
extern "C" int dpu_test(char *);
extern "C" int checksum(char *);

namespace py = pybind11;

/**
 * @brief Container class for interfacing with python
 *
 * This class holds data that can be reused
 * during different runs of the k-means algorithm.
 */
class Container
{
private:
    float *mean;
    int nfeatures;
    uint64_t npoints;
    uint64_t npadded;
    uint32_t ndpu;
    dpu_set allset;
    float scale_factor;
    float threshold;
    float **features_float;
    int_feature **features_int;
    bool from_file = false;

public:
    /**
     * @brief Loads binary into the DPUs
     *
     * @param DPU_BINARY Path to the binary.
     */
    void load_kernel(const char *DPU_BINARY)
    {
        ::load_kernel(&allset, DPU_BINARY, &ndpu);
    }
    /**
     * @brief Loads data into the DPU from a file.
     *
     * @param filename Path to the data file.
     * @param is_binary_file Whether the data is encoded as binary.
     */
    void load_file_data(const char *filename, bool is_binary_file, float threshold_in, int verbose)
    {
        from_file = true;
        if (is_binary_file)
            read_bin_input(filename, &npoints, &npadded, &nfeatures, ndpu, &features_float);
        else
            read_txt_input(filename, &npoints, &npadded, &nfeatures, ndpu, &features_float);
            save_dat_file(filename, npoints, nfeatures, features_float);
        transfer_data(threshold_in, verbose);
    }
    /**
     * @brief Loads data into the DPUs from a python array
     *
     * @param data A python ndarray.
     * @param npoints_in Number of points.
     * @param nfeatures_in Number of features.
     */
    void load_array_data(py::array_t<float> data, uint64_t npoints_in, int nfeatures_in, float threshold_in, int verbose)
    {
        float *data_ptr = (float *)data.request().ptr;

        npoints = npoints_in;
        nfeatures = nfeatures_in;
        format_array_input(npoints, &npadded, nfeatures, ndpu, data_ptr, &features_float);
        transfer_data(threshold_in, verbose);
    }
    /**
     * @brief Preprocesses and transfers quantized data to the DPUs
     *
     */
    void transfer_data(float threshold_in, int verbose)
    {
        threshold = threshold_in;
        scale_factor = preprocessing(&mean, nfeatures, npoints, npadded, features_float, &features_int, &threshold, verbose);
        populateDpu(features_int, nfeatures, npoints, npadded, ndpu, &allset);
    }
    /**
     * @brief Frees the data.
     */
    void free_data(bool from_file)
    {
        /* We are NOT freeing the underlying float array if it is managed by python */
        if (from_file)
            free(features_float[0]);
        else
            postprocessing(npoints, nfeatures, features_float, mean);
        free(features_float);
        free(features_int[0]);
        free(features_int);
        free(mean);
    }
    /**
     * @brief Frees the DPUs
     */
    void free_dpus()
    {
        ::free_dpus(&allset);
    }

    py::array_t<float> kmeans_cpp(
        int max_nclusters,
        int min_nclusters,
        int isRMSE,
        int isOutput,
        int nloops,
        py::array_t<int> log_iterations,
        py::array_t<double> log_time);
};

/**
 * @brief Main function for the KMeans algorithm.
 *
 * @return py::array_t<float> The centroids coordinates found by the algorithm.
 */
py::array_t<float> Container ::kmeans_cpp(
    int max_nclusters,               /**< upper bound of the number of clusters */
    int min_nclusters,               /**< lower bound of the number of clusters */
    int isRMSE,                      /**< whether or not RMSE is computed */
    int isOutput,                    /**< whether or not to print the centroids */
    int nloops,                      /**< how many times the algorithm will be executed for each number of clusters */
    py::array_t<int> log_iterations, /**< array logging the iterations per nclusters */
    py::array_t<double> log_time)    /**< array logging the time taken per nclusters */
{
    int best_nclusters = max_nclusters;

    int *log_iter_ptr = (int *)log_iterations.request().ptr;
    double *log_time_ptr = (double *)log_time.request().ptr;

    float *clusters = kmeans_c(
        features_float,
        features_int,
        nfeatures,
        npoints,
        npadded,
        scale_factor,
        threshold,
        mean,
        max_nclusters,
        min_nclusters,
        isRMSE,
        isOutput,
        nloops,
        log_iter_ptr,
        log_time_ptr,
        ndpu,
        &allset,
        &best_nclusters);

    std::vector<ssize_t> shape = {best_nclusters, nfeatures};
    std::vector<ssize_t> strides = {(int)sizeof(float) * nfeatures, sizeof(float)};

    py::capsule free_when_done(clusters, [](void *f)
                               { delete reinterpret_cast<float *>(f); });

    return py::array_t<float>(
        shape,
        strides,
        clusters,
        free_when_done);
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = R"pbdoc(
        DPU kmeans plugin
        -----------------

        .. currentmodule:: dpu_kmeans

        .. autosummary::
           :toctree: _generate

           add
           subtract
           call_home
           dpu_test
           checksum
           kmeans_cpp
    )pbdoc";

    py::class_<Container>(m, "Container", R"pbdoc(
        Container object to interface with the DPUs
    )pbdoc")
        .def(py::init<>())
        .def("load_kernel", &Container::load_kernel)
        .def("load_array_data", &Container::load_array_data)
        .def("free_data", &Container::free_data)
        .def("free_dpus", &Container::free_dpus)
        .def("kmeans", &Container::kmeans_cpp);

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j)
        { return i - j; },
        R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("call_home", &call_home, R"pbdoc(
        Get a number from the c file
    )pbdoc");

    m.def("dpu_test", &dpu_test, R"pbdoc(
        Call hello world on dpu
    )pbdoc");

    m.def("checksum", &checksum, R"pbdoc(
        Checksum test on dpus
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
