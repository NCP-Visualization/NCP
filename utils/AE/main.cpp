#include <iostream>
#include <fstream>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <omp.h>
#include <queue>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef _MSC_VER
    #include <intrin.h>
    #include <nmmintrin.h>
    #define __builtin_popcountll _mm_popcnt_u64
#endif
#define PI 3.14159265
#define ITERATIONS 10000
#define DUMMY_PENALTY 1000000
#define DUMMY_BASE 1e12
#define OFFSET 12
const int baseRd = 13331;
typedef unsigned long long uLL;

bool sortbysec(const std::pair<double*, double> &a, const std::pair<double*, double> &b) {
    return a.second < b.second;
}

bool sortbysec_rev(const std::pair<int, double> &a, const std::pair<int, double> &b) {
    return a.second > b.second;
}


class AESolver {
private:
    int N;
    std::vector<double> radii;
    std::vector<std::vector<double>> offsets;
    std::vector<std::pair<double*, double>> grids;
    std::vector<std::vector<int>> gridNeighbors;
    std::unordered_set<uLL> neighbors_set;
    std::vector<std::pair<int, int>> searchPairs;
    std::vector<std::vector<double>> dist_matrix;

    int threadNum;


private:

    void generate_grids(int size) {
        int layer = 0, index = 0;
        bool compute_new_grids = true;
        double* tmp;
        double farthest = -1;
        while (compute_new_grids) {
            if (layer == 0) {
                ++index;
                tmp = new double[2];
                tmp[0] = tmp[1] = 0;
                grids.push_back(std::make_pair(tmp, 0.0));
                gridNeighbors.push_back(std::vector<int>());
            } else {
                double min_squared_d = 1e9;
                for (int i = 0; i < 6; ++i) {
                    double start_x = cos(PI * i / 3) * layer,
                           start_y = sin(PI * i / 3) * layer,
                           end_x = cos(PI * (i + 1) / 3) * layer,
                           end_y = sin(PI * (i + 1) / 3) * layer;
                    for (int j = 0; j < layer; ++j) {
                        double x = start_x + (end_x - start_x) * j / layer,
                               y = start_y + (end_y - start_y) * j / layer,
                               squared_d = pow(x, 2) + pow(y, 2);
                        if (farthest < 0 || squared_d < farthest) {
                            ++index;
                            tmp = new double[2];
                            tmp[0] = x;
                            tmp[1] = y;
                            grids.push_back(std::make_pair(tmp, squared_d));
                            gridNeighbors.push_back(std::vector<int>());
                            if (squared_d < min_squared_d) {
                                min_squared_d = squared_d;
                            }
                        }
                    }
                }
                if (index >= size) {
                    if (farthest < 0) {
                        farthest = pow(layer + 3, 2);
                    }  else if (min_squared_d > farthest) {
                        compute_new_grids = false;
                    }
                }
            }
            ++layer;
        }

        std::sort(grids.begin(), grids.end(), sortbysec);

        // double neighborDist = 1.1, searchDist = 10.1;
        double neighborDist = 1.1, searchDist = 1e9;

        for (int i = 0; i < grids.size(); ++i) {
            for (int j = i + 1; j < grids.size(); ++j) {
                double d = pow(grids[i].first[0] - grids[j].first[0], 2) + pow(grids[i].first[1] - grids[j].first[1], 2);
                if (d < searchDist) {
                    searchPairs.push_back(std::make_pair(i, j));
                    if (d < neighborDist) {
                        gridNeighbors[i].push_back(j);
                        gridNeighbors[j].push_back(i);
                        neighbors_set.insert((i << OFFSET) + j);
                        neighbors_set.insert((j << OFFSET) + i);
                    }
                }
            }
        }

    }

public:
    AESolver() {
        threadNum = omp_get_num_procs();
        // std::cout << "Thread Num: " << threadNum << std::endl;
    }
    ~AESolver(){
        for (auto grid: grids) {
            delete grid.first;
        }
    }
    
    void setRadii(const std::vector<double>& _radii) {
        radii = _radii;
        N = radii.size();
    }

    void setDist(const std::vector<std::vector<double>> & dist) {
        dist_matrix = dist;
    }

    std::vector<int>* solveSingleFast(int len, std::vector<std::vector<double>>* cost) {
        // Initialize feasible solution
        int penalty = 1000000;
        int threadId = omp_get_thread_num();
        auto flag = new int[N];
        int currFlag = baseRd * (rand() & 0xffff);

        int* sol = new int[grids.size()];
        memset(sol, -1, sizeof(int) * grids.size());
        int* occupied = new int[grids.size()]; // -1 for available; 0 for point; 1 for glyph sides
        memset(occupied, -1, sizeof(int) * grids.size());

        int valid_grid_count;
        int idx = 0;
        double min_cost = -1;
        for (int i = 0; i < len; ++i) {
            double this_cost = 0;
            for (int j = 0; j < len; ++j) {
                this_cost += (*cost)[i][j];
            }
            if (this_cost < min_cost || min_cost == -1) {
                min_cost = this_cost;
                idx = i;
            }
        }
        flag[idx] = currFlag;
        sol[0] = idx;
        occupied[0] = 0;

        int count = 1;

        for (int k = 0; count < len && k < grids.size(); ++k) {
            if (occupied[k] >= 0) continue;
            min_cost = -1;
            idx = -1;
            for (int i = 0; i < len; ++i) {
                if (flag[i] == currFlag) continue;
                double this_cost = 0;
                for (int j: gridNeighbors[k]) {
                    // if (j >= k) break;
                    if (sol[j] == -1) {
                        this_cost += penalty;
                    }
                    else {
                        this_cost += (*cost)[i][sol[j]];
                    }
                }
                if (this_cost < min_cost || min_cost == -1) {
                    min_cost = this_cost;
                    idx = i;
                }
            }
            if (min_cost != -1) {
                flag[idx] = currFlag;
                occupied[k] = 0;
                sol[k] = idx;
                ++count;
            }
        }

        int grid_idx = 0;
        for (int i = 0; i < len; ++i) {
            bool placed = false;
            if (flag[i] == currFlag) continue;
            while (grid_idx < grids.size() && !placed) {
                ++grid_idx;
                if (sol[grid_idx] != -1) continue;
                flag[i] = currFlag;
                occupied[grid_idx] = 0;
                sol[grid_idx] = i;
                ++count;
                placed = true;
            }
        }

        double max_grid_dist = 0;
        for (int k = 0; k < grids.size(); ++k) {
            if (sol[k] != -1) {
                valid_grid_count = k;
                max_grid_dist = grids[k].second;
            }
        }
        while (valid_grid_count < grids.size() && grids[valid_grid_count].second < pow(sqrt(grids[valid_grid_count].second) + 5, 2)) {
            ++valid_grid_count;
        }

        // Swapping
        double* nn_cost = new double[grids.size()];
        for (int i = 0; i < valid_grid_count; ++i) {
            int k = sol[i], t;
            double sum = 0;
            for (int j: gridNeighbors[i]) {
                t = sol[j];
                if (k == -1 || t == -1) {
                    sum += penalty;
                } else {
                    sum += (*cost)[k][t];
                }
            }
            nn_cost[i] = sum;
        }
        int index = 0;
        std::vector<std::pair<int, double>> candidates;
        while (index < searchPairs.size()) {
            auto cand = searchPairs[index];
            if (cand.first >= valid_grid_count) {
                break;
            }
            if (cand.second < valid_grid_count) {
                int i = cand.first, j = cand.second;
                double before = nn_cost[i] + nn_cost[j], after = 0;
                for (int k: gridNeighbors[i]) {
                    if (k >= valid_grid_count) break;
                    if (sol[j] == -1 || sol[k] == -1) {
                        after += penalty;
                    } else {
                        after += (*cost)[sol[j]][sol[k]];
                    }
                }
                for (int k: gridNeighbors[j]) {
                    if (k >= valid_grid_count) break;
                    if (sol[i] == -1 || sol[k] == -1) {
                        after += penalty;
                    } else {
                        after += (*cost)[sol[i]][sol[k]];
                    }
                }
                if (neighbors_set.find((i << OFFSET) + j) != neighbors_set.end()) {
                    if (sol[i] == -1 || sol[j] == -1) {
                        after += 2 * penalty;
                    } else {
                        after += 2 * (*cost)[sol[i]][sol[j]];
                    }
                }
                double dummy = 0;
                if (sol[i] != -1 && sol[j] == -1) {
                    dummy = DUMMY_BASE / (grids[j].second - grids[i].second);
                } else if (sol[j] != -1 && sol[i] == -1) {
                    dummy = DUMMY_BASE / (grids[i].second - grids[j].second);
                }
                candidates.push_back(std::make_pair(index, after - before + dummy));
            }
            ++index;
        }

        int iter = 0;
        while (iter++ < ITERATIONS) {
            int min_cand_idx = -1;
            double min_delta = 0;
            for (auto& cand: candidates) {
                if (cand.second < min_delta) {
                    min_delta = cand.second;
                    min_cand_idx = cand.first;
                }
            }
            if (min_cand_idx == -1) break;
            int s = searchPairs[min_cand_idx].first,
                t = searchPairs[min_cand_idx].second;
            sol[s] ^= sol[t] ^= sol[s] ^= sol[t];
            // Update nn_cost
            std::unordered_set<int> seeds;
            seeds.insert(s);
            seeds.insert(t);
            for (int k: gridNeighbors[s]) {
                if (k < valid_grid_count)
                    seeds.insert(k);
            }
            for (int k: gridNeighbors[t]) {
                if (k < valid_grid_count)
                    seeds.insert(k);
            }
            for (int i: seeds) {
                double sum = 0;
                for (int j: gridNeighbors[i]) {
                    if (j >= valid_grid_count) break;
                    if (sol[i] == -1 || sol[j] == -1) {
                        sum += penalty;
                    } else {
                        sum += (*cost)[sol[i]][sol[j]];
                    }
                }
                nn_cost[i] = sum;
            }
            // Update candidates
            for (auto& cand: candidates) {
                int i = searchPairs[cand.first].first, j = searchPairs[cand.first].second;
                if (seeds.find(i) == seeds.end() && seeds.find(j) == seeds.end()) continue;
                double before = nn_cost[i] + nn_cost[j], after = 0;
                for (int k: gridNeighbors[i]) {
                    if (k >= valid_grid_count) break;
                    if (sol[j]  == -1 || sol[k]  == -1) {
                        after += penalty;
                    } else {
                        after += (*cost)[sol[j]][sol[k]];
                    }
                }
                for (int k: gridNeighbors[j]) {
                    if (k >= valid_grid_count) break;
                    if (sol[i] == -1 || sol[k]  == -1) {
                        after += penalty;
                    } else {
                        after += (*cost)[sol[i]][sol[k]];
                    }
                }
                if (neighbors_set.find((i << OFFSET) + j) != neighbors_set.end()) {
                    if (sol[i]  == -1 || sol[j] == -1) {
                        after += 2 * penalty;
                    } else {
                        after += 2 * (*cost)[sol[i]][sol[j]];
                    }
                }
                double dummy = 0;
                if (sol[i] != -1 && sol[j] == -1) {
                    dummy = DUMMY_BASE / (grids[j].second - grids[i].second);
                } else if (sol[j] != -1 && sol[i] == -1) {
                    dummy = DUMMY_BASE / (grids[i].second - grids[j].second);
                }
                cand.second = after - before + dummy;
            }
        }
        // Convert row asses to col asses
        std::vector<int>* ret = new std::vector<int>(len);

        int farthest_grid_idx = -1;
        for (int k = 0; k < grids.size(); ++k) {
            if (occupied[k] >= 0) farthest_grid_idx = k;
            if (occupied[k] > 0) continue;
            if (sol[k] != -1) {
                (*ret)[sol[k]] = k;
            }
        }

        delete flag;
        delete nn_cost;
        delete sol;
        delete occupied;
        return ret;
    }

    std::vector<std::vector<double>> layout() {
        generate_grids(N);
        int *tmp = new int[N];
        offsets.clear();
        auto asses = solveSingleFast(N, & dist_matrix);
        for (int j = 0; j < N; ++j) {
            tmp[j] = asses->at(j);
        }
        delete asses;
        for (int i = 0; i < N; ++i) {
            std::vector<double> pos;
            pos.push_back(grids[tmp[i]].first[0]);
            pos.push_back(grids[tmp[i]].first[1]);
            offsets.push_back(pos);
        }
        delete tmp;
        return offsets;
    }

    std::vector<std::vector<double>> getPositions() {
        float d = (sqrt(grids.back().second) + 0.5) * 2;
        std::vector<std::vector<double>> ret;
        for (int i = 0; i < N; ++i) {
            std::vector<double> pos, offset;
            offset = offsets[i];
            pos.push_back(offset[0] / d);
            pos.push_back(offset[1] / d);
            ret.push_back(pos);
        }
        return ret;
    }

    std::vector<double> getRadii() {
        float d = (sqrt(grids.back().second) + 0.5) * 2;
        std::vector<double> ret;
        for (int i = 0; i < N; ++i) {
            ret.push_back(0.5 / d);
        }
        return ret;
    }


};

PYBIND11_MODULE(AEPacking, m){
    m.doc() = "pybind11 example";
    pybind11::class_<AESolver>(m, "AESolver")
        .def( pybind11::init() )
        .def( "setRadii", &AESolver::setRadii )
        .def( "setDist", &AESolver::setDist )
        .def( "layout", &AESolver::layout )
        .def( "getPositions", &AESolver::getPositions )
        .def( "getRadii", &AESolver::getRadii );
}
