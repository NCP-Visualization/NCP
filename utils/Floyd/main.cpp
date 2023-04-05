#include <vector>
#include <tuple>
#include <list>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

class Floyd
{
public:
    int n;
    vector<vector<double>> d;
    void set_d(vector<vector<double>> d)
    {
        this->d = d;
    }
    void set_n(int n)
    {
        this->n = n;
    }
    void run()
    {
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (d[i][k] + d[k][j] < d[i][j])
                    {
                        d[i][j] = d[i][k] + d[k][j];
                    }
                }
            }
        }
    }
    vector<vector<double>> get_d()
    {
        return d;
    }
};

PYBIND11_MODULE(Floyd, m)
{
    m.doc() = "pybind11 example";
    pybind11::class_<Floyd>(m, "Floyd")
        .def(pybind11::init())
        .def("run", &Floyd::run)
        .def("set_d", &Floyd::set_d)
        .def("set_n", &Floyd::set_n)
        .def("get_d", &Floyd::get_d);
}

