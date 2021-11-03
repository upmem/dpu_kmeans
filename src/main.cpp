#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
// #include <wordexp.h>

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
extern "C" float *kmeans_c(float *, const char *, int, int, float, int, int, int, int, int, const char *, const char *, int *, int *);

namespace py = pybind11;

py::array_t<float> kmeans_cpp(
    py::array_t<float> data, /**< array holding the points */
    const char *filename,    /**< path of the data file */
    bool fileInput,          /**< whether the input is in a file */
    int isBinaryFile,        /**< whether or not the data file is serialized */
    float threshold,         /**< threshold for termination of the algorithm */
    int max_nclusters,       /**< upper bound of the number of clusters */
    int min_nclusters,       /**< lower bound of the number of clusters */
    int isRMSE,              /**< whether or not RMSE is computed */
    int isOutput,            /**< whether or not to print the centroids */
    int nloops,              /**< how many times the algorithm will be executed for each number of clusters */
    const char *DPU_BINARY,  /**< path to the dpu kernel */
    const char *log_name
    // py::array_t<float> testarray
)
{
    int ndim = 2;
    int best_nclusters = max_nclusters, nfeatures;

    float *data_ptr = (float *) data.request().ptr;
    float *clusters = kmeans_c(
        data_ptr,
        filename,
        fileInput,
        isBinaryFile,
        threshold,
        max_nclusters,
        min_nclusters,
        isRMSE,
        isOutput,
        nloops,
        DPU_BINARY,
        log_name,
        &best_nclusters,
        &nfeatures
        );

    // int n = sizeof(testarray);
    // std::cerr << "n: " << n << std::endl;
    // auto buf = testarray.request(true);
    // float* ptr =(float*) buf.ptr;
    // for (int i = 0; i < 25; i++)
    // {
    //     ptr[i] = (float)i;
    // }

    std::vector<ssize_t> shape = {best_nclusters, nfeatures};
    std::vector<ssize_t> strides = {(int)sizeof(float) * nfeatures, sizeof(float)};

    py::capsule free_when_done(clusters, [](void *f) {
        delete reinterpret_cast<float *>(f);
    });

    return py::array_t<float>(
        shape,
        strides,
        clusters,
        free_when_done
    );
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

    m.def("kmeans_cpp", &kmeans_cpp, R"pbdoc(
        Main kmeans function in c++
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
