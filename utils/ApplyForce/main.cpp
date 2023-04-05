#include <vector>
#include <tuple>
#include <list>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


using namespace std;

class ApplyForce
{
public:
    int n;
    vector<vector<double>> gravity;
    vector<vector<double>> attraction;
    vector<vector<double>> pos;
    vector<double> rad;
    vector<vector<double>> attraction_pairs;

    void set_rad(vector<double> rad)
    {
        this->rad = rad;
    }

    void set_pos(vector<vector<double>> pos)
    {
        this->pos = pos;
    }

    void set_n(int n)
    {
        this->n = n;
    }

    void set_attraction_pairs(vector<vector<double>> attraction_pairs)
    {
        this->attraction_pairs = attraction_pairs;
    }

    vector<vector<double>> get_gravity(double gravity_mag, double size_mag)
    {
        gravity = vector<vector<double>>(n, vector<double>(2, 0));
        double centroid = 0.5 * size_mag;
        // #pragma omp parallel for
        for (int i = 0; i < n; i++){
            double x = -pos[i][0] + centroid, y = -pos[i][1] + centroid;
            double norm = sqrt(x * x + y * y);
            x /= norm;
            y /= norm;
            // // FDP
            // norm = norm*norm;
            // direction[0] *= norm;
            // direction[1] *= norm;
            // // FA2
            // direction[0] *= norm;
            // direction[1] *= norm;
            // LinLog

            // // SM: F =  2||x_i-x_j||/d^2_ij * e_ij
            // // d_ij denotes the graph distance between nodes i and j
            // // so d_ij is 1 and SM == FA2
            // direction[0] *= norm;
            // direction[1] *= norm;

            // // MARS: F = 2||x_i-x_j||/d^2_ij * e_ij
            // // so d_ij is 1 and MARS == FA2
            // direction[0] *= norm;
            // direction[1] *= norm;

            // // SSM: F = 2||x_i-x_j||/d^2_ij * e_ij
            // // Maxent: F = 2||x_i-x_j||/d^2_ij * e_ij

            gravity[i][0] = x * gravity_mag;
            gravity[i][1] = y * gravity_mag;
        }
        return gravity;
    }

    vector<vector<double>> get_attraction(double attraction_mag)
    {
        attraction = vector<vector<double>>(n, vector<double>(2, 0));
        // #pragma omp parallel for
        for (int i = 0; i < attraction_pairs.size(); i++){
            int idx1 = attraction_pairs[i][0];
            int idx2 = attraction_pairs[i][1];
            double x = pos[idx2][0]-pos[idx1][0], y = pos[idx2][1]-pos[idx1][1];
            double norm = sqrt(x * x + y * y);
            x /= norm;
            y /= norm;
            // double r = rad[idx1]+rad[idx2];

            // // FDP
            // norm = norm*norm;
            // direction[0] *= norm;
            // direction[1] *= norm;
            // FA2
            // direction[0] *= norm;
            // direction[1] *= norm;
            // // LinLog



            // // SM: F =  2||x_i-x_j||/d^2_ij * e_ij
            // // d_ij denotes the graph distance between nodes i and j
            // // so d_ij is 1 and SM == FA2
            // direction[0] *= norm;
            // direction[1] *= norm;

            // // MARS: F = 2||x_i-x_j||/d^2_ij * e_ij
            // // so d_ij is 1 and MARS == FA2
            // direction[0] *= norm;
            // direction[1] *= norm;

            // // SSM: F = 2||x_i-x_j||/d^2_ij * e_ij
            // // Maxent: F = 2||x_i-x_j||/d^2_ij * e_ij

            // double mag = 2*(norm-r)/pow(r,2) * attraction_pairs[i][2]*attraction_mag;
            
            // #pragma omp critical
            {
                attraction[idx1][0] += x * attraction_mag;
                attraction[idx1][1] += y * attraction_mag;
                attraction[idx2][0] -= x * attraction_mag;
                attraction[idx2][1] -= y * attraction_mag;
            }
        }
        return attraction;
    }
};

PYBIND11_MODULE(ApplyForce, m)
{
    m.doc() = "pybind11 example";
    pybind11::class_<ApplyForce>(m, "ApplyForce")
        .def(pybind11::init())
        .def("set_n", &ApplyForce::set_n)
        .def("set_rad", &ApplyForce::set_rad)
        .def("set_pos", &ApplyForce::set_pos)
        .def("set_attraction_pairs", &ApplyForce::set_attraction_pairs)
        .def("get_gravity", &ApplyForce::get_gravity)
        .def("get_attraction", &ApplyForce::get_attraction);
}

