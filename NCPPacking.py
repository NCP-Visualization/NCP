import math
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score
from utils.Voronoi import Voronoi
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon, Point
import time
import pickle
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed
from CGAL.CGAL_Kernel import *
from CGAL.CGAL_Triangulation_2 import *
from CGAL import CGAL_Convex_hull_2
from utils.config import config
from scipy.spatial.distance import cdist
from utils.b2d.b2d import Box2DSimulator, get_concave
from utils.EPD import EPD
from matplotlib import pyplot as plt
import ctypes



# Rect Boundary
xl, yb, xr, yt = 0, 0, 1, 1
boundary_l = (1, 0, -xl)
boundary_r = (1, 0, -xr)
boundary_b = (0, 1, -yb)
boundary_t = (0, 1, -yt)
boundary_rect = {
    "segments": [boundary_l, boundary_t, boundary_r, boundary_b],
    "type": 'rect'
}

def convexhull_area(positions, radii, selected_indices=None):
    if selected_indices is None:
        positions = np.array(positions)
        radii = np.array(radii)
    else:
        positions = np.array(positions)[selected_indices]
        radii = np.array(radii)[selected_indices]
    N = len(radii)
    APPROX_NUM = 50
    approx_points = []
    for i in range(N):
        pos = positions[i]
        r = radii[i]
        for i in range(APPROX_NUM):
            approx_points.append(
                [pos[0] + r * np.cos(np.pi * 2 * i / APPROX_NUM), pos[1] + r * np.sin(np.pi * 2 * i / APPROX_NUM)])
    convex_hull = ConvexHull(approx_points)
    # get the area of the convex_hull
    return convex_hull.volume


def get_high_dimensional_clustering(dataset):
    # get the high dimensional clustering result
    data_path = os.path.join(config.data_root, 'high_dimensional_clustering/' + dataset + '-kmeans.pkl')
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['cluster_labels']
    else:
        raise ValueError("Could not find hd clustering buffer")


def single_cell_optimization_force_sep(k, cluster_pos, cluster_r, cluster_cell, extra_params, r0, gamma=0):
    t1=time.time()
    cluster_hull = cluster_cell
    cluster_center = cellCentroid(cluster_cell)
    iterations = extra_params['top_iters']


    n = len(cluster_pos)
    epd_instance = EPD.EPD()
    epd_instance.set_n(n)
    cells, original_flowers = computePowerDiagramByCGAL(cluster_pos, cluster_r ** 2, hull=cluster_hull,
                                                        return_flower=True)
    original_flower_indices = [[j for _, j in flower if j < n] for flower in original_flowers]

    epd_instance.set_cluster_pos(cluster_pos)
    epd_instance.set_cluster_r(cluster_r)
    epd_instance.set_flowers(original_flower_indices)
    epd_instance.set_center(cluster_center)
    epd_instance.set_gamma(gamma)

    tri_pairs = [[] for _ in range(n)]
    for i, flower in enumerate(original_flowers):
        lenf = len(flower)
        for j in range(lenf):
            ia, ib, ic = flower[j][-1], flower[(j + 1) % lenf][-1], flower[(j + 2) % lenf][-1]
            if ia >= n or ib >= n or ic >= n:
                continue
            if i > ib:
                continue
            tup = [i, ia, ib, ic]
            tri_pairs[i].append(tup)
            tri_pairs[ia].append(tup)
            tri_pairs[ib].append(tup)
            tri_pairs[ic].append(tup)

    epd_instance.set_tri_pairs(tri_pairs)

    for _ in range(iterations):
        ti0 = time.time()
        cells = computePowerDiagramByCGAL(cluster_pos, cluster_r ** 2, hull=cluster_hull, return_flower=False)

        epd_instance.set_cells(cells)

        flag = epd_instance.iter()
        if not flag:
            break
        cluster_pos = np.array(epd_instance.get_cluster_pos())
        cluster_r = np.array(epd_instance.get_cluster_r())

  
    t2 = time.time()
    print(t2-t1, t1)

    _, _, pairs = computePowerDiagramBruteForce(positions=cluster_pos, weights=cluster_r ** 2, radii=cluster_r,
                                                clipHull=True, hull=cluster_cell)
    r_scale = cluster_r[0] / r0
    return r_scale, pairs, k, cluster_pos, cluster_r


def cross_product(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]


def rotateClockwise(vec):
    return np.array([vec[1], -vec[0]])


def rotateCounterClockwise(vec):
    return np.array([-vec[1], vec[0]])


def rotateByOrigin(points, angle):
    rad = angle * math.pi / 180
    rotation_matrix = np.array([[math.cos(rad), -math.sin(rad)],
                                [math.sin(rad), math.cos(rad)]])
    return np.dot(rotation_matrix, points.T).T


def isCocurrent(v1, v2):
    return np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2) > 0


def modularity(vec):
    return np.sqrt(np.sum(vec ** 2))


def normalize(vec):
    return vec / modularity(vec)


def intersect(line1, line2, return_invalid=False):
    (a1, b1, c1) = line1
    (a2, b2, c2) = line2
    tmp = a1 * b2 - a2 * b1
    if abs(tmp) < 1e-12:
        return None
    else:
        x = (c2 * b1 - c1 * b2) / tmp
        y = (a2 * c1 - a1 * c2) / tmp
        if return_invalid or (xl <= x <= xr and yb <= y <= yt):
            return np.array([x, y])
        else:
            return None


def dis(p, ln):
    return -(ln[0] * p[0] + ln[1] * p[1] + ln[2]) / math.sqrt(ln[0] ** 2 + ln[1] ** 2)


def cellArea(cell):
    if isinstance(cell, Polygon):
        try:
            return cell.area
        except:
            return 0

    area = 0
    for k in range(len(cell)):
        p1 = cell[k]
        p2 = cell[(k + 1) % len(cell)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    area /= 2
    return abs(area)


def cellOrientation(cell):
    area = 0
    for k in range(len(cell)):
        p1 = cell[k]
        p2 = cell[(k + 1) % len(cell)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    return area > 0


def cellCentroid(cell):
    if isinstance(cell, Polygon):
        try:
            return np.array([cell.centroid.x, cell.centroid.y])
        except:
            return None

    x, y = 0, 0
    area = 0
    for k in range(len(cell)):
        p1 = cell[k]
        p2 = cell[(k + 1) % len(cell)]
        v = p1[0] * p2[1] - p2[0] * p1[1]
        area += v
        x += (p1[0] + p2[0]) * v
        y += (p1[1] + p2[1]) * v
    area *= 3
    if area == 0:
        return None
    return np.array([x / area, y / area])


def cellInscribedCircleRadius(cell, site):
    if isinstance(cell, Polygon):
        cell = np.array(cell.exterior.coords[:-1])
    if len(cell) == 0:
        return 0
    r = 1e9
    cell = np.array(cell)
    for k in range(len(cell)):
        p1 = cell[k]
        p2 = cell[(k + 1) % len(cell)]
        edgeLength = np.sqrt(np.sum((p1 - p2) ** 2))
        if edgeLength < 1e-12:
            continue
        v = cross_product(p1 - site, p2 - site)
        r = min(r, abs(v / edgeLength))
    return r


def computePowerDiagramBruteForce(positions, weights=None, radii=None, clipHull=False, hull=None, dummy=None,
                                  vsol=None):
    N = positions.shape[0]
    if weights is None:
        weights = np.zeros(N)

    if dummy is not None:
        M = dummy.shape[0]
        positions = np.vstack([positions, dummy])
        weights = np.hstack([weights, np.zeros(M)])
        radii = np.hstack([radii, np.zeros(M)])

    if vsol is None:
        vsol = Voronoi.Voronoi()
    vsol.clearSites()
    vsol.inputSites(positions)
    vsol.setWeight(weights)
    if radii is not None and clipHull:
        vsol.setRadius(radii)

    if hull is not None:
        vsol.setBoundary(hull)

    if clipHull:
        res = vsol.computeBruteForce(True)
    else:
        res = vsol.computeBruteForce(False)

    boundary_return = None
    if clipHull and hull is None:
        boundary_return = {
            'hull': np.array(vsol.getConvexHullVertices()),
            'edges': np.array(vsol.getConvexHullEdges()),
            'sites': np.array(vsol.generateBoundarySites())
        }

    cells, flowers = [], []
    for (cell, flower) in res:
        cells.append(np.array(cell))
        flowers.append(np.array(flower))

    preserved_pairs = [(i, j) for i in range(N) for j in flowers[i] if i < j < N]

    if boundary_return is not None:
        return cells, flowers, preserved_pairs, boundary_return
    else:
        return cells, flowers, preserved_pairs


def computePowerDiagramByCGAL(positions, weights=None, hull=None, return_flower=False):
    if weights is None:
        nonneg_weights = np.zeros(len(positions))
    else:
        nonneg_weights = weights - np.min(weights)

    rt = Regular_triangulation_2()

    v_handles = []
    v_handles_mapping = {}
    k = 0

    # ti0 = time.time()

    for pos, w in zip(positions, nonneg_weights):
        v_handle = rt.insert(Weighted_point_2(Point_2(float(pos[0]), float(pos[1])), float(w)))
        v_handles.append(v_handle)
        v_handles_mapping[v_handle] = k
        k += 1

    control_point_set = [
        Weighted_point_2(Point_2(-10, -10), 0),
        Weighted_point_2(Point_2(10, -10), 0),
        Weighted_point_2(Point_2(10, 10), 0),
        Weighted_point_2(Point_2(-10, 10), 0)
    ]

    for cwp in control_point_set:
        v_handle = rt.insert(cwp)
        v_handles.append(v_handle)
        v_handles_mapping[v_handle] = k
        k += 1

    # ti1 = time.time()
    # print(ti1 - ti0, positions.shape, "pd-iter")

    poly_cells = []
    cells = []
    flowers = []

    # for i, handle in enumerate(v_handles):
    i = 0
    tt1, tt2, tt3 = 0, 0, 0
    for handle in rt.finite_vertices():
        t = time.time()
        non_hidden_point = handle.point()
        x, y = non_hidden_point.x(), non_hidden_point.y()
        while i < len(positions) and (x - positions[i][0]) ** 2 + (y - positions[i][1]) ** 2 > 1e-10:
            i += 1
            poly_cells.append([])
        if i >= len(positions):
            break

        f = rt.incident_faces(handle)
        done = f.next()
        cell = []
        while True:
            face_circulator = f.next()
            wc = rt.weighted_circumcenter(face_circulator)
            cell.append((wc.x(), wc.y()))
            if face_circulator == done:
                break
        tt1 += time.time() - t
        # ***************************************
        # poly_cell = Polygon(cell)
        # poly_cells.append(poly_cell)
        poly_cells.append(cell)
        # if hull is not None and not hull.contains(poly_cell):
        #     poly_cell = hull.intersection(poly_cell)
        # cells.append(poly_cell)
        #
        # tt2 += time.time() - t
        # **************************************

        if return_flower:
            v = rt.incident_vertices(handle)
            done = v.next()
            flower = []
            while True:
                vertex_circulator = v.next()
                p = vertex_circulator.point()
                flower.append(([p.x(), p.y(), p.weight()], v_handles_mapping[vertex_circulator]))
                if vertex_circulator == done:
                    break
            flowers.append(flower)

        tt3 += time.time() - t

        i += 1
    # ti2 = time.time()
    # print(ti2 - ti1, positions.shape, "pd-iter-2")
    # print(tt1, tt2, tt3, return_flower)

    # check_indices = []

    # hull = Polygon(hull)
    # for i, poly_cell in enumerate(poly_cells):
    #     c = Polygon(poly_cell)
    #     if hull is not None and not hull.contains(c):
    #         # check_indices.append(i)
    #         c = hull.intersection(c)
    #     cells.append(c)

    epd_instance = EPD.EPD()
    if not isinstance(hull, np.ndarray):
        hull = hull.exterior.coords[:-1]
    if hull is not None:
        cells = epd_instance.clipping(poly_cells, hull)
    else:
        cells = poly_cells

    tt2 += time.time() - t

    if return_flower:
        return cells, flowers
    else:
        return cells


def computeConvexHull(positions):
    point_set = []
    for pos in positions:
        point_set.append(Point_2(float(pos[0]), float(pos[1])))
    convex_hull = []
    CGAL_Convex_hull_2.convex_hull_2(point_set, convex_hull)
    cvp = [(v.x(), v.y()) for v in convex_hull]
    poly_hull = Polygon(cvp)
    return poly_hull


class NCPPacking(object):
    def __init__(self, data):
        self.data = data
        self.N = self.data['importance'].shape[0]
        self.arg_sim = np.argsort(-self.data['similarity'], axis=1)
        try:
            self.r = self.data['importance']
        except:
            self.r = [1 for _ in range(self.data['similarity'].shape[0])]

        if self.data['label'] is None:
            self.data['label'] = np.zeros(self.N, dtype=int)
        else:
            self.data['label'] = self.data['label'].astype(np.int)

        self.cluster = self.data['label']

        self.config = {
            "point-placement": "tSNE",
            "graph-building": "Delaunay",
            "optimization": "Sep-Force-H-CPD",
            "compaction": "Box2D"
        }
        self.debug = False
        self.logs = {}
        self.preserved_nn_pairs = []
        self.pre_layout_positions = None

        self.fixed_indices = []
        self.extra_params = {}
        self.extra_params['gravity'] = 1
        self.extra_params['utopian'] = True
        self.extra_params['iterations'] = 1250
        self.extra_params['rotation_angle'] = 45
        self.extra_params['gamma'] = 2
        self.extra_params['lambda'] = 2
        self.extra_params['convex_iters'] = 50
        self.extra_params['top_iters'] = 30
        self.extra_params['cpd_iters'] = 100
        self.extra_params['alpha_min'] = 0.0005
        self.extra_params['a'] = 2


        self.logger = None
        self.eval_intermediate = False

        self.intermediate_result = {}

        self.inter_pairs = None

    def __str__(self):
        idf = []
        for name in self.config:
            idf.append(self.config[name][0])
        ret = "3Phase" + "-" + "-".join(idf)
        extra_ret = ''
        for key in self.extra_params:
            if key not in ['top_iters', 'cpd_iters', 'alpha_min', 'rotation_angle', 'utopian']:
                extra_ret += '~' + key + '~' + str(self.extra_params[key])
        return ret + extra_ret

    def set_logger(self, logger):
        self.logger = logger

    def run(self):
        initial_pos = self.pre_layout()
        self.pre_layout_positions = initial_pos
        mask = self.build_graph(initial_pos)
        print("initialize done")

        t1 = time.time()
        intermediate_prefix = "-".join([self.config['point-placement'], str(self.extra_params['utopian']),
                                        self.config['optimization'], str(self.extra_params['rotation_angle']),
                                        str(self.extra_params['top_iters']), str(self.extra_params['gamma'])])
        if not os.path.exists(os.path.join(config.data_root, 'intermediate_cache', intermediate_prefix)):
            os.makedirs(os.path.join(config.data_root, 'intermediate_cache', intermediate_prefix))
        intermediate_cache = os.path.join(config.data_root, 'intermediate_cache', intermediate_prefix,
                                          f"{self.data['dataset_name']}.pkl")
        # if os.path.exists(intermediate_cache):
        if False:
            with open(intermediate_cache, "rb") as inter_f:
                (rough_circle_list, rough_links) = pickle.load(inter_f)
        else:
            rough_circle_list, rough_links = self.optimize(initial_pos, mask)
            # with open(intermediate_cache, "wb") as inter_f:
            #     pickle.dump((rough_circle_list, rough_links), inter_f)
        print("optimize done")

        t2 = time.time()
        print("optimize:", t2 - t1)
        fine_circle_list = self.post_process(rough_circle_list, rough_links)
        print("compaction done")
        t3 = time.time()
        print("compaction:", t3 - t2)
        
        positions = [[circle[0], circle[1]] for circle in fine_circle_list]
        radii = [circle[2] for circle in fine_circle_list]

        return positions, radii

    def pre_layout(self):
        cfg = self.config['point-placement']
        if 'utopian' in self.extra_params and self.extra_params['utopian']:
            projection_cache = os.path.join(config.data_root, "projection-utopian",
                                            f"{self.data['dataset_name']}-{cfg}.pkl")
        else:
            projection_cache = os.path.join(config.data_root, "projection", f"{self.data['dataset_name']}-{cfg}.pkl")
        if os.path.exists(projection_cache):
            with open(projection_cache, "rb") as f:
                projection_data = pickle.load(f)
                return projection_data['pos']

        if self.data['feature'] is not None:
            metric = 'euclidean'
            X = self.data['feature']
        else:
            assert cfg != "PCA"
            metric = 'precomputed'
            X = self.data['distance'] if self.data['distance'] is not None else (1 - self.data['similarity'])

        if 'utopian' in self.extra_params and self.extra_params['utopian']:
            metric = 'precomputed'
            cluster_data_path = os.path.join(config.data_root,
                                             'high_dimensional_clustering/' + self.data['dataset_name'] + '-kmeans.pkl')
            with open(cluster_data_path, 'rb') as f:
                cluster_data = pickle.load(f)
            cluster_labels = cluster_data['cluster_labels']
            X = self.data['distance']
            cluster_alpha = 0.5
            for i in np.unique(cluster_labels):
                cluster_indices = np.where(cluster_labels == i)[0]
                X[cluster_indices][:, cluster_indices] *= cluster_alpha

        if cfg == "Hierarchy":
            Y = np.zeros((len(self.r), 2))
        else:
            if cfg == "tSNE":
                model = TSNE(n_components=2, method='exact', metric=metric, perplexity=15, random_state=0,
                             early_exaggeration=6, n_iter=3000)
            elif cfg == "MDS":
                model = MDS(n_components=2, dissimilarity=metric, random_state=1)
            elif cfg == "PCA":
                model = PCA(n_components=2, random_state=0)
            else:
                raise NotImplementedError
            # short cut
            if X.shape[1] > 2:
                Y = model.fit_transform(X)
            else:
                Y = X

        Y = (Y - Y.min(axis=0)) / Y.ptp(axis=0)
        Y[:, 0] += (1 - np.max(Y[:, 0])) / 2
        Y[:, 1] += (1 - np.max(Y[:, 1])) / 2

        with open(projection_cache, "wb") as f:
            pickle.dump({
                "projection_model": model,
                "pos": Y
            }, f)

        return Y

    def build_graph(self, pos):
        cfg = self.config['graph-building']
        # mask = np.zeros((self.N, self.N))
        mask = []
        if cfg == "Delaunay":
            tri = Delaunay(pos)
            for (a, b, c) in tri.simplices:
                # mask[a, b] = mask[b, a] = mask[a, c] = mask[c, a] = mask[b, c] = mask[c, b] = 1
                mask.append((a, b))
                mask.append((b, c))
                mask.append((c, a))
        else:
            return None
        return mask

    def clustering(self, positions):
        """
        for clustering score ablation
        """
        if 'utopian' in self.extra_params and self.extra_params['utopian']:
            clustering_cache = os.path.join(config.data_root, "clustering-utopian",
                                            f"{self.data['dataset_name']}-kmeans.pkl")
        else:
            clustering_cache = os.path.join(config.data_root, "clustering", f"{self.data['dataset_name']}-kmeans.pkl")
        if os.path.exists(clustering_cache):
            with open(clustering_cache, "rb") as f:
                cluster_data = pickle.load(f)
                self.cluster = cluster_data['cluster_labels']
        else:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.metrics import calinski_harabasz_score
            from sklearn.metrics import davies_bouldin_score
            best_score = 0
            best_clustering_labels = None
            best_nc = -1
            for nc in range(2, 10):
                clustering_model = KMeans(n_clusters=nc, random_state=0)
                clustering_model.fit(positions)
                clustering_labels = clustering_model.predict(positions)
                score = silhouette_score(positions, clustering_labels)
                # score = calinski_harabasz_score(positions, clustering_labels)
                # score = davies_bouldin_score(positions, clustering_labels)
                print(nc, score)
                if score > best_score:
                    best_score = score
                    best_clustering_labels = clustering_labels
                    best_nc = nc

            with open(clustering_cache, "wb") as f:
                pickle.dump({
                    "clustering_model": clustering_model,
                    "nc": best_nc,
                    "cluster_labels": best_clustering_labels
                }, f)

            self.cluster = best_clustering_labels

    def optimize(self, initial_pos, mask):
        cfg = self.config['optimization']

        def grad_W(cells, caps):
            grad = []
            for cell, cap in zip(cells, caps):
                area = cellArea(cell)
                grad.append(cap - area)
            return np.array(grad)

        def grad_X(positions, caps, cells):
            grad = []
            for i, cell in enumerate(cells):
                centroid = cellCentroid(cell)
                area = cellArea(cell)
                if centroid is not None:
                    # grad.append(2 * area * np.array(centroid - positions[i]))
                    grad.append(np.array(centroid - positions[i]))
                else:
                    grad.append(np.array([0, 0]))
            return np.array(grad)

        def F(cells, sites, caps, weights):
            ret = 0
            for i, cell in enumerate(cells):
                area = cellArea(cell)
                cx, cy = sites[i, 0], sites[i, 1]
                if area > 0:
                    centroid = cellCentroid(cell)
                    ret += -2 * (cx * centroid[0] + cy * centroid[1]) * area + area * (cx ** 2 + cy ** 2)
                ret -= weights[i] * (area - caps[i])
            return ret

        def find_W(positions, w0, caps, hull):
            # Maximize F
            half = 0.5
            # For L-BFGS
            m = 5

            cells = computePowerDiagramByCGAL(positions, w0, hull=hull)

            H0 = np.eye(len(w0))
            s, y, rho = [], [], []
            k = 1
            gk = -grad_W(cells, caps)
            dk = -H0.dot(gk)
            current_F = F(cells, positions, caps, w0)
            count = 0
            while True:
                count += 1
                n = 0
                mk = -1
                gk = -grad_W(cells, caps)
                if modularity(gk) < 1e-12:
                    break
                # dk = normalize(dk)
                while n < 20:
                    new_w = w0 + (half ** n) * dk
                    new_cells = computePowerDiagramByCGAL(positions, new_w, hull=hull)
                    if len(new_cells) == len(positions):
                        new_F = F(new_cells, positions, caps, new_w)
                    else:
                        new_F = np.nan

                    if not np.isnan(new_F) and new_F > current_F:
                        mk = n
                        break
                    n += 1
                if mk < 0:
                    break
                w = new_w
                cells = new_cells
                current_F = new_F

                sk = w - w0
                qk = -grad_W(cells, caps)
                yk = qk - gk
                s.append(sk)
                y.append(yk)
                rho.append(1 / yk.T.dot(sk))

                a = []
                for i in range(max(k - m, 0), k):
                    alpha = rho[k - i - 1] * s[k - i - 1].T.dot(qk)
                    qk = qk - alpha * y[k - i - 1]
                    a.append(alpha)
                r = H0.dot(qk)
                for i in range(max(k - m, 0), k):
                    beta = rho[i] * y[i].T.dot(r)
                    r = r + s[i] * (a[k - i - 1] - beta)

                if rho[-1] > 0:
                    dk = -r

                k += 1
                w0 = w

            return w0

        def find_transform(cell, cell_center, points):
            point_hull = points[ConvexHull(points).vertices]
            if self.extra_params['center'] == 'point':
                points_center = np.mean(points, axis=0)
            else:
                points_center = cellCentroid(point_hull)
            shifted_points = point_hull + cell_center - points_center
            poly_points = Polygon(shifted_points)
            if isinstance(cell, Polygon):
                poly_cell = cell
            else:
                poly_cell = Polygon(cell)
            scaling_min, scaling_max = 1, 1  # min containt, max not containt
            s = 1
            if poly_cell.contains(poly_points):
                while True:
                    s *= 2
                    new_shifted_points = (shifted_points - cell_center) * s + cell_center
                    new_poly = Polygon(new_shifted_points)
                    if poly_cell.contains(new_poly):
                        continue
                    else:
                        scaling_max = s
                        scaling_min = s / 2
                        break
            else:
                while True:
                    s /= 2
                    new_shifted_points = (shifted_points - cell_center) * s + cell_center
                    new_poly = Polygon(new_shifted_points)
                    if not poly_cell.contains(new_poly):
                        continue
                    else:
                        scaling_max = s * 2
                        scaling_min = s
                        break
            while scaling_max - scaling_min > 1e-6:
                scaling_mid = (scaling_max + scaling_min) / 2
                new_shifted_points = (shifted_points - cell_center) * scaling_mid + cell_center
                new_poly = Polygon(new_shifted_points)
                if not poly_cell.contains(new_poly):
                    scaling_max = scaling_mid
                else:
                    scaling_min = scaling_mid

            best_scaling = scaling_min
            best_angle = 0

            for ang_size in range(self.extra_params['rotation_angle'], 0, -2):
                for ang_sign in [1, -1]:
                    ang = ang_size * ang_sign
                    rotated_shifted_points = rotateByOrigin(shifted_points - cell_center, ang) + cell_center
                    s = best_scaling
                    while True:
                        new_shifted_points = (rotated_shifted_points - cell_center) * s + cell_center
                        new_poly = Polygon(new_shifted_points)
                        if poly_cell.contains(new_poly):
                            s *= 1.1
                            continue
                        else:
                            scaling_max = s
                            scaling_min = s / 1.1
                            break
                    if s <= best_scaling:
                        continue
                    else:
                        while scaling_max - scaling_min > 1e-6:
                            scaling_mid = (scaling_max + scaling_min) / 2
                            new_shifted_points = (shifted_points - cell_center) * scaling_mid + cell_center
                            new_poly = Polygon(new_shifted_points)
                            if not poly_cell.contains(new_poly):
                                scaling_max = scaling_mid
                            else:
                                scaling_min = scaling_mid
                        best_scaling = scaling_min
                        best_angle = ang
            return best_scaling, best_angle

        if cfg == "Sep-Force-H-CPD":
            self.extra_params['center'] = 'hull'
            self.extra_params['update_hull'] = True

            margin = 0.05
            pos = initial_pos * (1 - margin * 2) + margin
            N = self.r.shape[0]
            dis_mat = cdist(pos, pos, metric='euclidean')
            np.fill_diagonal(dis_mat, np.inf)
            r_mat = np.tile(self.r, N).reshape((N, N))
            r_mat += r_mat.T
            ratio = min(np.min(dis_mat / r_mat), margin / np.max(self.r))
            radii = self.r * ratio
            self.clustering(pos)

            cluster_indices = []
            for clabel in np.unique(self.cluster):
                selected_indices = np.where(self.cluster == clabel)[0]
                cluster_indices.append(selected_indices)
                # Scaling by cluster
                N = selected_indices.shape[0]
                dis_mat = cdist(pos[selected_indices], pos[selected_indices], metric='euclidean')
                np.fill_diagonal(dis_mat, np.inf)
                r_mat = np.tile(self.r[selected_indices], N).reshape((N, N))
                r_mat += r_mat.T
                ratio = np.min(dis_mat / r_mat)
                radii[selected_indices] *= ratio

            hull = computeConvexHull(pos)

            top_level_iterations = 100

            score_list = []
            top_level_res = []

            score_list.append(silhouette_score(pos, self.cluster, metric='euclidean'))

            hull = None
            for _ in range(top_level_iterations):

                cluster_centers = []
                cluster_weights = []
                cluster_radii = []

                if hull is None or self.extra_params['update_hull']:
                    # Update Boundary Hull
                    hull = computeConvexHull(pos)

                for k in range(len(np.unique(self.cluster))):
                    selected_indices = cluster_indices[k]
                    if self.extra_params['center'] == 'point':
                        cluster_center = pos[selected_indices].mean(axis=0)
                    else:
                        cpos = pos[selected_indices]
                        cluster_center = cellCentroid(cpos[ConvexHull(cpos).vertices])
                    cluster_centers.append(cluster_center)
                    cluster_radii.append(np.max(np.sqrt(np.sum((pos[selected_indices] - cluster_center) ** 2, axis=1))))
                    cluster_weight = np.sum(self.r[selected_indices] ** 2)
                    cluster_weights.append(cluster_weight)

                cluster_centers = np.array(cluster_centers)
                cluster_radii = np.array(cluster_radii)
                cluster_weights = np.array(cluster_weights)

                cc = cluster_centers.copy()

                capacity = cluster_weights / np.sum(cluster_weights) * cellArea(hull)
                w = find_W(cluster_centers, np.zeros(len(cluster_centers)), capacity, hull)
                cells = computePowerDiagramByCGAL(positions=cluster_centers, weights=w, hull=hull)

                for i, cell in enumerate(cells):
                    center = cellCentroid(cell)
                    radius = cellInscribedCircleRadius(cell, center)
                    cluster_centers[i] = center
                    cluster_radii[i] = radius

                cluster_scale_ratio = 1e10
                for k in range(len(np.unique(self.cluster))):
                    selected_indices = cluster_indices[k]
                    max_scaling_ratio, max_scaling_rotation_angle = find_transform(cells[k], cluster_centers[k],
                                                                                   pos[selected_indices])
                    pos[selected_indices] = rotateByOrigin(pos[selected_indices] - cc[k],
                                                           max_scaling_rotation_angle) * max_scaling_ratio + \
                                            cluster_centers[k]
                    radii[selected_indices] *= max_scaling_ratio
                    cluster_scale_ratio = min(max_scaling_ratio, cluster_scale_ratio)

                top_level_res.append((pos.copy(), radii.copy(), cells))
                new_score = silhouette_score(pos, self.cluster, metric='euclidean')

                if len(score_list) > 0 and new_score > score_list[-1]:
                    break
                else:
                    score_list.append(new_score)

            pos, radii, cells = top_level_res[-1]
            margin = 0.1
            resolution = max(np.max(pos[:, 0] + radii) - np.min(pos[:, 0] - radii),
                             np.max(pos[:, 1] + radii) - np.min(pos[:, 1] - radii)) + 2 * margin

            offset_x = (np.max(pos[:, 0] + radii) + np.min(pos[:, 0] - radii)) / 2
            offset_y = (np.max(pos[:, 1] + radii) + np.min(pos[:, 1] - radii)) / 2
            offset = np.array([offset_x, offset_y])
            cluster_centers -= offset
            cluster_centers /= resolution
            cluster_centers += 0.5
            cluster_radii /= resolution
            pos = (pos - offset) / resolution + 0.5
            radii /= resolution

            cluster_cells = []
            for c in cells:
                if isinstance(c, Polygon):
                    cell = np.array(c.exterior.coords[:-1])
                else:
                    cell = c
                rescaled_cell = (np.array(cell) - offset) / resolution + 0.5
                cluster_cells.append(rescaled_cell)

            # ******************************
            MultiNum = len(cluster_indices)
            params = []

            # default
            # gamma = 0.05
            gamma = 1.5
            if 'gamma' in self.extra_params:
                gamma = self.extra_params['gamma']

            for k in range(len(np.unique(self.cluster))):
                selected_indices = cluster_indices[k]
                cluster_pos = pos[selected_indices]
                cluster_r = radii[selected_indices]
                ext_prm = self.extra_params.copy()
                ext_prm['dataset_name'] = self.data['dataset_name']
                param = (k, cluster_pos, cluster_r, cluster_cells[k], ext_prm, self.r[selected_indices][0], gamma)
                params.append(param)

            opt = single_cell_optimization_force_sep

            res = []
            pool = ThreadPool(MultiNum)
            res = pool.starmap(opt, params)
            pool.close()
            pool.join()

            pairs = []
            min_scale = 1e9
            for it in res:
                min_scale = min(min_scale, it[0])
                selected_indices = cluster_indices[it[2]]
                pos[selected_indices] = it[3]
                radii[selected_indices] = it[4]
                pairs += [(selected_indices[j], selected_indices[k]) for (j, k) in it[1]]

            radii = self.r * min_scale
            circle_list = [[pos[i, 0], pos[i, 1], radii[i]] for i in range(self.N)]
            preservation = set(pairs)
        else:
            raise NotImplementedError

        return circle_list, preservation

    def post_process(self, circle_list, preservation):
        time_0 = time.time()
        cfg = self.config['compaction']
        initial_pos = np.array([c[:2] for c in circle_list])
        radii = np.array([c[-1] for c in circle_list])

        MAGNITUDE = 100
        if cfg == "Box2D":
            config_b2d = {
                "size_mag": 1.0,
                "iterations": self.extra_params['iterations'],
                "attraction_mag": self.extra_params['gravity'] * self.extra_params['gamma'] * MAGNITUDE,
                "convexity_mag": self.extra_params['gravity'] * self.extra_params['lambda'] * MAGNITUDE,
                "gravity": self.extra_params['gravity'] * MAGNITUDE,
                "density": 0,
                "bullet": False,
                "time_step": 0.005,
                "convexity_iters": self.extra_params['convex_iters']
            }

            flag = False # True -> py
            if config_b2d['iterations'] > 5000:
                flag = True
            alpha = 1
            alpha_min = self.extra_params['alpha_min']
            alpha_decay = (1 - (alpha_min ** (1 / config_b2d["iterations"])))
            alpha_target = 0
            decay = -alpha_decay * (alpha_target - alpha)
            if flag:
                alpha_decay = (1 - (alpha_min ** (1 / config_b2d["iterations"])))
                alpha_target = 0

            if flag:
                attractions = [(i, j, self.data['similarity'][i, j]) for (i, j) in preservation if
                               self.cluster[i] == self.cluster[j]]
            else:
                attractions = [(i, j) for (i, j) in preservation if
                            self.cluster[i] == self.cluster[j]]

            pre_pos = initial_pos.copy()
            rad = radii.copy() * self.extra_params['a']

            config_b2d["size_mag"] = 1 / np.min(rad)
            print("mag", config_b2d['size_mag'])
            if 'size_mag' in self.extra_params:
                config_b2d["size_mag"] = self.extra_params['size_mag']

            config_b2d["gravity"] *= config_b2d["size_mag"]
            config_b2d["attraction_mag"] *= config_b2d["size_mag"]
            config_b2d["convexity_mag"] *= config_b2d["size_mag"]
            if not flag:
                cluster_label_mapping = {}
                for i, cluster_label in enumerate(np.unique(self.cluster)):
                    cluster_label_mapping[cluster_label] = i
                new_cluster = np.array([cluster_label_mapping[c] for c in self.cluster])
                groups = []
                for i in range(len(cluster_label_mapping)):
                    groups.append(np.where(new_cluster == i)[0])

                n_iters = config_b2d["iterations"]
                # load box2d.dll from the same directory as this file
                box2d = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'box2d_parallel.dll'))
                res = np.zeros((2 * len(pre_pos)), dtype=np.float64)
                pre_pos *= config_b2d["size_mag"]
                rad *= config_b2d["size_mag"]
                time_1 = time.time()
                for fd_iter in range(0, n_iters, config_b2d['convexity_iters']):
                    if fd_iter + config_b2d['convexity_iters'] > n_iters:
                        config_b2d['convexity_iters'] = n_iters - fd_iter
                    # print("pre_pos: ", pre_pos)
                    # print("rad: ", rad)
                    # print("fd_iter: ", fd_iter)
                    # print("alpha: ", alpha)
                    time_10 = time.time()
                    n_hull, hull = get_concave(pre_pos, rad, groups)

                    time_11 = time.time()
                    c_pre_pos = pre_pos.flatten()
                    c_pre_pos = c_pre_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    c_rad = rad.flatten()
                    c_rad = c_rad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    c_n_hull = np.array(n_hull, dtype=np.int32)
                    c_n_hull = c_n_hull.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                    c_hull = np.array(hull, dtype=np.int32)
                    c_hull = c_hull.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                    c_size_mag = ctypes.c_double(config_b2d["size_mag"])
                    c_gravity = ctypes.c_double(config_b2d["gravity"])
                    c_attraction_mag = ctypes.c_double(config_b2d["attraction_mag"])
                    c_convexity_mag = ctypes.c_double(config_b2d["convexity_mag"])
                    c_convexity_iters = ctypes.c_int(config_b2d["convexity_iters"])
                    c_alpha = ctypes.c_double(alpha)
                    c_decay = ctypes.c_double(decay)
                    c_attractions = np.array(attractions, dtype=np.int32)
                    c_attractions = c_attractions.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                    c_res = res.flatten()
                    c_res = c_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    time_11 = time.time()
                    box2d.Simulate(len(pre_pos), c_pre_pos, c_rad, len(n_hull), c_n_hull, c_hull, len(attractions),
                                     c_attractions, c_size_mag, c_gravity, c_attraction_mag, c_convexity_mag,
                                        c_convexity_iters, c_alpha, c_decay, c_res)
                    res = np.ctypeslib.as_array(c_res, shape=(len(pre_pos),2))
                    pre_pos = res
                    for j in range(config_b2d["convexity_iters"]):
                        alpha += alpha_decay * (alpha_target - alpha)
                    time_12 = time.time()
                    print("time per iter: ", time_12 - time_10)
                    print("time for convexity: ", time_11 - time_10)
                    # print(alpha)
            if flag:
                optimizer = Box2DSimulator(positions=pre_pos * config_b2d["size_mag"], radii=rad * config_b2d["size_mag"],
                                           cluster=self.cluster,
                                           size_mag=config_b2d["size_mag"],
                                           attractions=attractions, attraction_magnitude=config_b2d["attraction_mag"],
                                           convexity_magnitude=config_b2d["convexity_mag"],
                                           density=config_b2d["density"],
                                           gravity=config_b2d["gravity"], bullet=config_b2d["bullet"],
                                           time_step=config_b2d["time_step"])

                for fd_iter in range(config_b2d["iterations"]):
                    optimizer.clear_velocities()
                    forces = optimizer.apply_attractions_and_gravity(fd_iter % config_b2d['convexity_iters'] == 0)
                    optimizer.step()
                    alpha += alpha_decay * (alpha_target - alpha)
                    optimizer.attraction_magnitude = alpha * config_b2d["attraction_mag"]
                    optimizer.convexity_magnitude = alpha * config_b2d["convexity_mag"]
                    optimizer.gravity = alpha * config_b2d["gravity"]
                    if alpha <= alpha_min:
                        break
                pre_pos = np.array(optimizer.get_positions())
                rad = np.array(optimizer.get_radii())
            time_2 = time.time()
            # print("Box2D init time: ", time_1 - time_0)
            # print("Box2D time: ", time_2 - time_1)
            pos = pre_pos
            rad = rad

            resolution = max(np.max(pos[:, 0] + rad) - np.min(pos[:, 0] - rad),
                             np.max(pos[:, 1] + rad) - np.min(pos[:, 1] - rad))
            pos = (pos - (pos - rad.reshape(-1, 1)).min(axis=0)) / resolution
            rad /= resolution
            circle_list = [[pos[i, 0], pos[i, 1], rad[i]] for i in range(len(pos))]
            # pass
            pos = np.array([c[:2] for c in circle_list])
            rad = np.array([c[-1] for c in circle_list])
            circle_list = [[pos[i, 0], pos[i, 1], rad[i]] for i in range(len(pos))]
        else:
            raise NotImplementedError

        return circle_list
