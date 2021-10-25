#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

int array_sum(){
    return 0;
}

extern "C" char* call_home(char*);
extern "C" int dpu_test(char*);
extern "C" int checksum(char*);

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        DPU trees plugin
        ----------------

        .. currentmodule:: dpu_trees

        .. autosummary::
           :toctree: _generate

           add
           subtract
           call_home
           dpu_test
           checksum
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
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
