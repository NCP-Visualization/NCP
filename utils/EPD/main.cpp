#include <vector>
#include <tuple>
#include <list>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py=pybind11;
using namespace std;
#define min(a, b) (a)<(b)?(a):(b)

class EPD
{
public:
    int n;
    vector<vector<double>> cluster_pos;
    vector<double> cluster_r;
    vector<vector<int>> original_flowers;
    vector<double> cluster_center;
    double scaling = 1;
    double ga;
    vector<vector<double>> main_direction;
    vector<vector<double>> centroids;
    vector<vector<vector<int>>> tri_pairs;
    vector<vector<vector<double>>> cells;

    void set_n(int n)
    {
        this->n = n;
    }


    void set_cluster_pos(vector<vector<double>> &cluster_pos) {
        this->cluster_pos = cluster_pos;
    }

    void set_cluster_r(vector<double> &cluster_r) {
        this->cluster_r = cluster_r;
    }

    void set_flowers(vector<vector<int>> &flowers) {
        this->original_flowers = flowers;
    }

    void set_cells(vector<vector<vector<double>>> &cells) {
        this->cells = cells;
    }

    void set_tri_pairs(vector<vector<vector<int>>> &tri_pairs) {
        this->tri_pairs = tri_pairs;
    }

    void set_center(vector<double> &cluster_center) {
        this->cluster_center = cluster_center;
    }

    void set_gamma(double ga) {
        this->ga = ga;
    }

    double loss(double scale, vector<vector<double>> &pos, vector<double> &center, vector<vector<int>> &flowers, double gamma) {
        double U_gravity = 0;
        for (int i = 0; i < n; ++i) {
            U_gravity += sqrt((pos[i][0] - center[0])*(pos[i][0] - center[0]) + (pos[i][1] - center[1]) * (pos[i][1] - center[1]));
        }
        U_gravity /= scale;
        // np.sum(np.linalg.norm(positions - center, axis=1)) / scale

        double U_attraction = 0;

        for (int i = 0; i < n; ++i) {
            for (int j : flowers[i]) {
                U_attraction += sqrt((pos[i][0] - pos[j][0])*(pos[i][0] - pos[j][0]) + (pos[i][1] - pos[j][1]) * (pos[i][1] - pos[j][1]));
            }
        }
        U_attraction /= scale;

        return U_gravity + gamma * U_attraction;
    }

    double loss_part(double scale, vector<double> &this_pos, vector<double> &center, vector<vector<double>> &flower_pos, double gamma) {
        double U_gravity = sqrt((this_pos[0] - center[0])*(this_pos[0] - center[0]) + (this_pos[1] - center[1]) * (this_pos[1] - center[1])) / scale;
        double U_attraction = 0;
        for (auto &pos: flower_pos) {
            U_attraction += sqrt((this_pos[0] - pos[0])*(this_pos[0] - pos[0]) + (this_pos[1] - pos[1]) * (this_pos[1] - pos[1]));
        }
        U_attraction *= 2;
        U_attraction /= scale;

        return U_gravity + gamma * U_attraction;
    }

    double loss_part_i(double scale, vector<vector<double>> &pos, vector<double> &this_pos, vector<double> &center, vector<int> flower, double gamma) {
        double U_gravity = sqrt((this_pos[0] - center[0])*(this_pos[0] - center[0]) + (this_pos[1] - center[1]) * (this_pos[1] - center[1])) / scale;
        double U_attraction = 0;
        for (int j : flower) {
            U_attraction += sqrt(
                (this_pos[0] - pos[j][0]) * (this_pos[0] - pos[j][0]) +
                (this_pos[1] - pos[j][1]) * (this_pos[1] - pos[j][1])
            );
        }
        U_attraction *= 2;
        U_attraction /= scale;

        return U_gravity + gamma * U_attraction;
    }


//    double loss() {
//        double U_gravity = 0;
//        for (int i = 0; i < n; ++i) {
//            U_gravity += sqrt((pos[i][0] - center[0])*(pos[i][0] - center[0]) + (pos[i][1] - center[1]) * (pos[i][1] - center[1]));
//        }
//        U_gravity /= scale;
//        // np.sum(np.linalg.norm(positions - center, axis=1)) / scale
//
//        double U_attraction = 0;
//
//        for (int i = 0; i < n; ++i) {
//            for (int j : flowers[i]) {
//                U_attraction += sqrt((pos[i][0] - pos[j][0])*(pos[i][0] - pos[j][0]) + (pos[i][1] - pos[j][1]) * (pos[i][1] - pos[j][1]));
//            }
//        }
//        U_attraction /= scale;
//
//        return U_gravity + gamma * U_attraction;
//    }
//
//    double loss_part(int i) {
//        vector<double> this_pos = pos[i]
//        double U_gravity = sqrt((this_pos[0] - center[0])*(this_pos[0] - center[0]) + (this_pos[1] - center[1]) * (this_pos[1] - center[1])) / scale;
//        double U_attraction = 0;
//        for (int j: flowers[i]) {
//            auto p = pos[j];
//            U_attraction += sqrt((this_pos[0] - p[0])*(this_pos[0] - p[0]) + (this_pos[1] - p[1]) * (this_pos[1] - p[1]));
//        }
//        U_attraction *= 2;
//        U_attraction /= scale;
//
//        return U_gravity + gamma * U_attraction;
//    }

    vector<double> compensate_dir(vector<double> a, vector<double> b, vector<double> c, vector<double> x, bool flag) {
//    vector<double> compensate_dir(vector<double> a, vector<double> b, vector<double> c, vector<double> x, bool flag) {
//        weighted_circumcenter_and_radius
//        vector<double> a = pos[ia], b = pos[ib], c = pos[ic], x = pos[ix];
//        a.push_back(weights[ia]);
//        b.push_back(weights[ib]);
//        c.push_back(weights[ic]);
//        x.push_back(weights[ix]);
        double A = ((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])) / 2;
        vector<double> e12, e13;
        e12.push_back(b[0]-a[0]); e12.push_back(b[1]-a[1]);
        e13.push_back(a[0]-c[0]); e13.push_back(a[1]-c[1]);
        double sq_len_12 = e12[0]*e12[0] + e12[1]*e12[1];
        double sq_len_13 = e13[0]*e13[0] + e13[1]*e13[1];
        double w13 = (sq_len_12 + a[2] - b[2]) / (4 * A);
        double w12 = (sq_len_13 + a[2] - c[2]) / (4 * A);
        vector<double> wc;
        wc.push_back(a[0] - w12 * e12[1] - w13 * e13[1]);
        wc.push_back(a[1] + w12 * e12[0] + w13 * e13[0]);
        double wr = (wc[0]-a[0])*(wc[0]-a[0]) + (wc[1]-a[1])*(wc[1]-a[1]) - a[2];
//        wc.push_back(wr);
//        return wc;
        double current_r = (wc[0]-x[0])*(wc[0]-x[0]) + (wc[1]-x[1])*(wc[1]-x[1]) - x[2];
        vector<double> dir(2, 0);

        if (flag && current_r > wr) {
            dir[0] = wc[0] - x[0];
            dir[1] = wc[1] - x[1];
            double norm = sqrt(dir[0]*dir[0]+dir[1]*dir[1]);
            dir[0] /= norm;
            dir[1] /= norm;
        } else if (!flag && current_r < wr) {
            dir[0] = x[0] - wc[0];
            dir[1] = x[1] - wc[1];
            double norm = sqrt(dir[0]*dir[0]+dir[1]*dir[1]);
            dir[0] /= norm;
            dir[1] /= norm;
        }
        return dir;
    }


    double cellInscribedCircleRadius(vector<vector<double>> cell, vector<double> site) {
        if (cell.size() == 0) return 0;
        double r = 1e9;
        for (int k = 0; k < cell.size(); ++k) {
            auto p1 = cell[k], p2 = cell[(k + 1) % cell.size()];
            double edgeLength = sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]));
            if (edgeLength < 1e-12) continue;
//            v = cross_product(p1 - site, p2 - site)
            double v = (p1[0]-site[0])*(p2[1]-site[1]) - (p1[1]-site[1])*(p2[0]-site[0]);
            r = min(r, abs(v / edgeLength));
        }
        return r;
    }

    vector<double> cellCentroid(vector<vector<double>> cell) {
        double x = 0, y = 0;
        double area = 0;
        for(int k = 0; k < cell.size(); k++) {
            auto p1 = cell[k];
            auto p2 = cell[(k + 1) % cell.size()];
            double v = p1[0] * p2[1] - p2[0] * p1[1];
            area += v;
            x += (p1[0] + p2[0]) * v;
            y += (p1[1] + p2[1]) * v;
        }
        area *= 3;
        vector<double> centroid;
        centroid.push_back(x / area);
        centroid.push_back(y / area);
        return centroid;
    }

    bool iter() {

        double last_energy = loss(scaling, cluster_pos, cluster_center, original_flowers, ga);
//        cout << last_energy << " cp" << endl;

        vector<vector<double>> best_pos;
        double best_ratio;
        double best_value;

        main_direction.clear();
        centroids.clear();
        vector<double> centroid_rad;
        vector<double> max_enlarge_scale(n, 0);

        for (int i = 0; i < n; ++i) {
            auto cell = cells[i];
            auto centroid = cellCentroid(cell);
            double max_rad = cellInscribedCircleRadius(cell, centroid);
            vector<double> md;
            md.push_back(centroid[0] - cluster_pos[i][0]);
            md.push_back(centroid[1] - cluster_pos[i][1]);
            main_direction.push_back(md);
            centroids.push_back(centroid);
            centroid_rad.push_back(max_rad);
        }

        bool flag = false;
        double main_direction_ratio = 1;
        best_value = last_energy;

        for (int j = 0; j < 5; ++j) {
            vector<vector<double>> new_pos;
            vector<double> new_rad;
            double ratio = 1e9;
            for (int i = 0; i < n; ++i) {
                vector<double> new_pos_i;
                new_pos_i.push_back(cluster_pos[i][0] + main_direction[i][0] * main_direction_ratio);
                new_pos_i.push_back(cluster_pos[i][1] + main_direction[i][1] * main_direction_ratio);
                new_pos.push_back(new_pos_i);
                new_rad.push_back(cellInscribedCircleRadius(cells[i], new_pos_i));
                max_enlarge_scale[i] = new_rad[i] / cluster_r[i];
                ratio = min(ratio, max_enlarge_scale[i]);
            }

            double move_to_centroid_value = loss(scaling * ratio, new_pos, cluster_center, original_flowers, ga);
//            cout << move_to_centroid_value << " cp1 " << j << endl;
            if (move_to_centroid_value < last_energy) {
                best_pos = new_pos;
                best_ratio = ratio;
                best_value = move_to_centroid_value;
//                best_weights = (cluster_r * ratio) ** 2;
                centroid_rad = new_rad;
                flag = true;
                break;
            } else {
                main_direction_ratio *= 0.9;
            }
        }


        if (!flag) {
            best_pos = cluster_pos;
            best_ratio = 1.0;
            best_value = last_energy;
//            best_weights = cluster_r ** 2;
        }

        vector<pair<double,int>> process_order;

        for (int i = 0; i < n ; i++) {
            // filling the original array
            process_order.push_back (make_pair(max_enlarge_scale[i], i)); // value, original index
        }
        sort(process_order.begin(), process_order.end());


        for (auto &t: process_order) {
            int i = t.second;
            vector<double> sub_direction(2, 0);
            for (auto &tp: tri_pairs[i]) {
                int ix = tp[0], ia = tp[1], ib = tp[2], ic = tp[3];
                vector<double> x(best_pos[ix]), a(best_pos[ia]), b(best_pos[ib]), c(best_pos[ic]);
                x.push_back(pow(cluster_r[ix] * best_ratio, 2));
                a.push_back(pow(cluster_r[ia] * best_ratio, 2));
                b.push_back(pow(cluster_r[ib] * best_ratio, 2));
                c.push_back(pow(cluster_r[ic] * best_ratio, 2));
                vector<double> dir;
                if (i == ix)
                    dir = compensate_dir(a, b, c, x, true);
                else if (i == ia)
                    dir = compensate_dir(b, c, x, a, false);
                else if (i == ib)
                    dir = compensate_dir(c, x, a, b, true);
                else if (i == ic)
                    dir = compensate_dir(x, a, b, c, false);
                sub_direction[0] += dir[0];
                sub_direction[1] += dir[1];
            }
            double norm = sqrt(sub_direction[0]*sub_direction[0] + sub_direction[1]*sub_direction[1]);
            if (norm > 0) {
                double offset_ratio = (centroid_rad[i] / 2) / norm;
                double last_value = loss_part_i(scaling * best_ratio, best_pos, best_pos[i], cluster_center, original_flowers[i], ga);
                vector<double> old_pos(best_pos[i]);
                double old_enlarge = max_enlarge_scale[i];

                max_enlarge_scale[i] = 1e10;
                double other_max_enlarge = *min_element(max_enlarge_scale.begin(), max_enlarge_scale.end());

                bool move_flag = false;

                for (int j = 0; j < 10; ++j) {
                    vector<double> new_pos(old_pos);
                    new_pos[0] += sub_direction[0] * offset_ratio;
                    new_pos[1] += sub_direction[1] * offset_ratio;
                    max_enlarge_scale[i] = cellInscribedCircleRadius(cells[i], new_pos) / cluster_r[i];
                    double ratio = min(max_enlarge_scale[i], other_max_enlarge);
                    double v = loss_part_i(scaling * ratio, best_pos, new_pos, cluster_center, original_flowers[i], ga);
                    if ((best_value - last_value) * (best_ratio / ratio) + v < best_value) {
                        best_pos[i] = new_pos;
                        best_ratio = ratio;
//                        best_weights = (cluster_r * ratio) ** 2
                        best_value = (best_value - last_value) * (best_ratio / ratio) + v;
                        flag = true;
                        move_flag = true;
//                        break;
                    } else {
                        offset_ratio /= 2;
                    }
                }

                if (!move_flag) {
                    max_enlarge_scale[i] = old_enlarge;
                }
            }
        }

//        cout << best_value << " " << best_ratio << " cp3" << endl;

        double new_ratio = 1.0;
        best_value = loss(scaling * best_ratio, best_pos, cluster_center, original_flowers, ga);
        if (best_ratio > 1.0) {
            for (int j = 0; j < 10; ++j) {
                new_ratio = (new_ratio + best_ratio) / 2;
                double v = loss(scaling * new_ratio, best_pos, cluster_center, original_flowers, ga);
                if (v < best_value) {
                    best_ratio = new_ratio;
                    best_value = v;
                    flag = true;
                    break;
                }
            }
        }

        if (!flag) return false;

        cluster_pos = best_pos;
        for (double &r: cluster_r) {
            r *= best_ratio;
        }
        scaling *= best_ratio;
        last_energy = best_value;
//        cout << cluster_r[0] << ' ' << cluster_pos[0][0] << ' ' << cluster_pos[0][1] << ' ' << last_energy << endl;
        return true;
    }

//    vector<vector<int>> find_perimeter(vector<pair<vector<int>, double>> alphasimplices, double alpha) {
//
//
//    }


    double circumradius(vector<int> tri) {
        auto a = cluster_pos[tri[0]], b = cluster_pos[tri[1]], c = cluster_pos[tri[2]];
        double four_S = abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])) * 2;
        double la = sqrt((b[0]-c[0])*(b[0]-c[0])+(b[1]-c[1])*(b[1]-c[1]));
        double lb = sqrt((a[0]-c[0])*(a[0]-c[0])+(a[1]-c[1])*(a[1]-c[1]));
        double lc = sqrt((b[0]-a[0])*(b[0]-a[0])+(b[1]-a[1])*(b[1]-a[1]));

        return la*lb*lc/four_S;
    }

    vector<int> concave_hull_indices(vector<std::pair<vector<int>, double>> alphasimplices, double alpha, int n) {
        std::unordered_set<int> edges;
        std::unordered_set<int> perimeter_edges;
        const int e[3][2] = {{0, 1}, {1, 2}, {0, 2}};
        for (auto & tri: alphasimplices) {
            auto point_indices = tri.first;
            double cr = tri.second;
            if (cr >= 1.0 / alpha) continue;
            sort(point_indices.begin(), point_indices.end());
            for (int i = 0; i < 3; ++i) {
                int v = point_indices[e[i][0]] * n + point_indices[e[i][1]];
                if (edges.find(v) == edges.end()) {
                    edges.insert(v);
                    perimeter_edges.insert(v);
                } else {
                    perimeter_edges.erase(v);
                }
            }
        }

        vector<int> ret;
        vector<vector<int>> adj(n, vector<int>());
        for (int v: perimeter_edges) {
            int i = v / n;
            int j = v % n;
            adj[i].push_back(j);
            adj[j].push_back(i);
        }

        int count = perimeter_edges.size();
        int this_ind = (*(perimeter_edges.begin())) / n;
        std::unordered_set<int> visited;
        while (ret.size() < count) {
            ret.push_back(this_ind);
            visited.insert(this_ind);
            bool flag = false;
            for (int k: adj[this_ind]) {
                if (visited.find(k) == visited.end()) {
                    flag = true;
                    this_ind = k;
                    break;
                }
            }
            if (!flag) break;
        }
//        cout << count << " peri num" << endl;
        if (ret.size() < count) {
            return vector<int>();
        } else {
            return ret;
        }
//        return ret;

    }

    vector<vector<double>> get_cluster_pos() {
        return cluster_pos;
    }

    vector<double> get_cluster_r() {
        return cluster_r;
    }


    std::vector<double> computeIntersection(std::vector<double> &v1, std::vector<double> &v2, std::tuple<double, double, double> &ln) {
        double a1 = std::get<0>(ln), b1 = std::get<1>(ln), c1 = std::get<2>(ln);
        double dx = v2[0] - v1[0], dy = v2[1] - v1[1];
        double a2 = -dy, b2 = dx, c2 = dy * v1[0] - dx * v1[1];
        double tmp = a1 * b2 - a2 * b1;
        double x = (c2 * b1 - c1 * b2) / tmp;
        double y = (a2 * c1 - a1 * c2) / tmp;
        std::vector<double> intersection;
        intersection.push_back(x);
        intersection.push_back(y);
        return intersection;
    }

    std::vector<double> computeIntersection(std::tuple<double, double, double> &ln1, std::tuple<double, double, double> &ln2) {
        double a1 = std::get<0>(ln1), b1 = std::get<1>(ln1), c1 = std::get<2>(ln1);
        double a2 = std::get<0>(ln2), b2 = std::get<1>(ln2), c2 = std::get<2>(ln2);
        double tmp = a1 * b2 - a2 * b1;
        double x = (c2 * b1 - c1 * b2) / tmp;
        double y = (a2 * c1 - a1 * c2) / tmp;
        std::vector<double> intersection;
        intersection.push_back(x);
        intersection.push_back(y);
        return intersection;
    }

    bool inside(std::vector<double> &p, std::tuple<double, double, double> &ln) {
        return std::get<0>(ln) * p[0] + std::get<1>(ln) * p[1] + std::get<2>(ln) > 0;
    }


    std::vector<std::vector<std::vector<double>>> clipping(std::vector<std::vector<std::vector<double>>> cells, std::vector<std::vector<double>> hull) {
        std::vector<std::tuple<double, double, double>> hull_segs;

        for (int i = 0; i < hull.size(); ++i) {
            double px = hull[i][0], py = hull[i][1], qx = hull[(i + 1) % hull.size()][0], qy = hull[(i + 1) % hull.size()][1];
            double a = py - qy, b = qx - px;
            double c = px * qy  - py * qx;
            hull_segs.push_back(std::make_tuple(a, b, c));
        }

        std::vector<std::vector<std::vector<double>>> ret;

        for (auto &cell : cells) {
            if (cell.size() == 0) {
                ret.push_back(std::vector<std::vector<double>>());
                continue;
            }
            std::vector<std::vector<double>> outputList(cell);
            for (auto &seg: hull_segs) {
                auto inputList(outputList);
                outputList.clear();
                auto s = inputList.back();
                for (int kk = 0; kk < inputList.size(); ++kk) {
                    auto e = inputList[kk];
                    if (inside(e, seg)) {
                        if (!inside(s, seg)) {
                            outputList.push_back(computeIntersection(s, e, seg));
                        }
                        outputList.push_back(e);
                    } else if (inside(s, seg)) {
                        outputList.push_back(computeIntersection(s, e, seg));
                    }
                    s = e;
                }
                if (outputList.size() < 3) {
                    ret.push_back(outputList);
                    break;
                }
            }
            ret.push_back(outputList);
        }
        return ret;
    }

    double compute_rad_ratio(std::vector<std::vector<double>> pos, std::vector<double> rad, std::vector<int> cluster) {
        int n = pos.size();
        double k = 1e9;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (cluster[i] != cluster[j]) continue;
                double dis = sqrt((pos[i][0]-pos[j][0])*(pos[i][0]-pos[j][0])+(pos[i][1]-pos[j][1])*(pos[i][1]-pos[j][1]));
//                cout << dis;
                k = min(k, dis/(rad[i]+rad[j]));
            }
        }
        return k;
    }

};

PYBIND11_MODULE(EPD, m)
{
    m.doc() = "pybind11 example";
    pybind11::class_<EPD>(m, "EPD")
        .def(pybind11::init())
        .def("set_n", &EPD::set_n, py::call_guard<py::gil_scoped_release>())
        .def("set_cluster_pos", &EPD::set_cluster_pos, py::call_guard<py::gil_scoped_release>())
        .def("set_cluster_r", &EPD::set_cluster_r, py::call_guard<py::gil_scoped_release>())
        .def("set_flowers", &EPD::set_flowers, py::call_guard<py::gil_scoped_release>())
        .def("set_center", &EPD::set_center, py::call_guard<py::gil_scoped_release>())
        .def("set_tri_pairs", &EPD::set_tri_pairs, py::call_guard<py::gil_scoped_release>())
        .def("set_cells", &EPD::set_cells, py::call_guard<py::gil_scoped_release>())
        .def("set_gamma", &EPD::set_gamma, py::call_guard<py::gil_scoped_release>())
        .def("loss", &EPD::loss, py::call_guard<py::gil_scoped_release>())
        .def("loss_part", &EPD::loss_part, py::call_guard<py::gil_scoped_release>())
        .def("compensate_dir", &EPD::compensate_dir, py::call_guard<py::gil_scoped_release>())
        .def("get_cluster_pos", &EPD::get_cluster_pos, py::call_guard<py::gil_scoped_release>())
        .def("get_cluster_r", &EPD::get_cluster_r, py::call_guard<py::gil_scoped_release>())
        .def("iter", &EPD::iter, py::call_guard<py::gil_scoped_release>())
        .def("circumradius", &EPD::circumradius, py::call_guard<py::gil_scoped_release>())
        .def("clipping", &EPD::clipping, py::call_guard<py::gil_scoped_release>())
        .def("concave_hull_indices", &EPD::concave_hull_indices, py::call_guard<py::gil_scoped_release>())
        .def("compute_rad_ratio", &EPD::compute_rad_ratio, py::call_guard<py::gil_scoped_release>())
        ;
}

//PYBIND11_MODULE(EPD, m)
//{
//    m.doc() = "pybind11 example";
//    pybind11::class_<EPD>(m, "EPD")
//        .def(pybind11::init())
//        .def("set_n", &EPD::set_n )
//        .def("set_cluster_pos", &EPD::set_cluster_pos )
//        .def("set_cluster_r", &EPD::set_cluster_r )
//        .def("set_flowers", &EPD::set_flowers )
//        .def("set_center", &EPD::set_center )
//        .def("set_tri_pairs", &EPD::set_tri_pairs )
//        .def("set_cells", &EPD::set_cells )
//        .def("set_gamma", &EPD::set_gamma )
//        .def("loss", &EPD::loss )
//        .def("loss_part", &EPD::loss_part )
//        .def("compensate_dir", &EPD::compensate_dir )
//        .def("get_cluster_pos", &EPD::get_cluster_pos )
//        .def("get_cluster_r", &EPD::get_cluster_r )
//        .def("iter", &EPD::iter )
//        .def("circumradius", &EPD::circumradius )
//        .def("clipping", &EPD::clipping )
//        .def("concave_hull_indices", &EPD::concave_hull_indices )
//        .def("compute_rad_ratio", &EPD::compute_rad_ratio )
//        ;
//}
