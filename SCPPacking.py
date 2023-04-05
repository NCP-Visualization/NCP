import math
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score
from utils.Voronoi import Voronoi
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon
import time
import pickle
from multiprocessing import Pool
from CGAL.CGAL_Kernel import *
from CGAL.CGAL_Triangulation_2 import *
from CGAL import CGAL_Convex_hull_2
from utils.config import config
from scipy.spatial.distance import cdist
from utils.b2d.b2d import Box2DSimulator


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

def single_cell_optimization(k, cluster_pos, cluster_r, cluster_cell, extra_params, r0):
    cluster_hull = Polygon(cluster_cell)
    iterations = extra_params['top_iters']
    for _ in range(iterations):
        cells = computePowerDiagramByCGAL(cluster_pos, cluster_r ** 2, hull=cluster_hull)
        # Relax radii and pos
        ratio = 1e5
        for i, cell in enumerate(cells):
            cell = np.array(cell.exterior.coords[:-1])
            if 'chebyshev' in extra_params and extra_params['chebyshev']:
                centroid, max_rad = cellInscribedCircleRadiusByChebyshev(cell)
            else:
                centroid = cellCentroid(cell)
                max_rad = cellInscribedCircleRadius(cell, centroid)
            cluster_pos[i] = centroid
            ratio = min(ratio, max_rad / cluster_r[i])
        cluster_r = cluster_r * ratio
    _, _, pairs = computePowerDiagramBruteForce(positions=cluster_pos, weights=cluster_r ** 2, radii=cluster_r,
                                                clipHull=True, hull=cluster_cell)
    r_scale = cluster_r[0] / r0
    return r_scale, pairs, k, cluster_pos, cluster_r


def cross_product(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]


def rotateByOrigin(points, angle):
    rad = angle * math.pi / 180
    rotation_matrix = np.array([[math.cos(rad), -math.sin(rad)],
                                [math.sin(rad), math.cos(rad)]])
    return np.dot(rotation_matrix, points.T).T


def modularity(vec):
    return np.sqrt(np.sum(vec ** 2))


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


def cellInscribedCircleRadiusByChebyshev(cell):
    """
    Calculate the center and radius of the maximum inscribed circle of an
    arbitrary convex polygon in 2D space
    :param points: a list of points coordinates which are the vertices of the polygon
    :return: the center and radius of the maximum inscribed circle
    """
    def cal_lines_by_points(points, d, center):
        """
        Calculate the parameters a, b, c of the line ax+by+c=0
        :param points: a list of points coordinates which are the vertices of the
        polygon
        :return: a list of lines
        """
        lines = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            a, b, c = cal_abc_from_line(p1[0], p1[1], p2[0], p2[1])
            c1 = c - d * np.sqrt(a * a + b * b)
            c2 = c + d * np.sqrt(a * a + b * b)
            d1 = abs(a * center[0] + b * center[1] + c1)
            d2 = abs(a * center[0] + b * center[1] + c2)
            if (d1 < d2):
                c = c1
            else:
                c = c2
            lines.append([a, b, c])
        return lines

    def cal_abc_from_line(x1, y1, x2, y2):
        """
        Calculate the parameters a, b, c of the line ax+by+c=0
        :param x1, y1: coordinate of the first point
        :param x2, y2: coordinate of the second point
        :return: a, b, c
        """
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        return a, b, c

    # calculate the distance between point p and line segment ab
    def distance_point_line(p, a, b):
        """
        Calculate the distance between point p and line segment ab
        :param p: vector point p
        :param a: vector point a
        :param b: vector point b
        :return: the distance between point p and line segment ab
        """
        pa = p - a
        ba = b - a
        h = np.dot(pa, ba) / np.dot(ba, ba)
        return np.linalg.norm(pa - h * ba)

    # calculate the min distance between point p and polygon
    def distance_point_polygon(p, polygon):
        """
        Calculate the min distance between point p and polygon
        :param p: vector point p
        :param polygon: a list of points coordinates which are the vertices of the
        polygon
        :return: the min distance between point p and polygon
        """
        min_distance = math.inf
        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            distance = distance_point_line(p, a, b)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    # select a point inside the polygon
    def center_point_inside_polygon(polygon):
        """
        Select the center point inside the polygon
        :param polygon: a list of points coordinates which are the vertices of the
        polygon
        :return: the center point inside the polygon
        """
        center = cellCentroid(polygon)
        return center

    def cal_new_polygon(polygon, segments):
        def _clip(subjectPolygon, clipSegs):
            def inside(p, ln):
                return ln[0] * p[0] + ln[1] * p[1] + ln[2] < 0

            def computeIntersection(v1, v2, ln):
                (a1, b1, c1) = ln
                dx, dy = v2[0] - v1[0], v2[1] - v1[1]
                a2, b2, c2 = -dy, dx, dy * v1[0] - dx * v1[1]
                tmp = a1 * b2 - a2 * b1
                x = (c2 * b1 - c1 * b2) / tmp
                y = (a2 * c1 - a1 * c2) / tmp
                return [x, y]

            outputList = subjectPolygon

            for seg in clipSegs:
                inputList = outputList
                outputList = []
                s = inputList[-1]
                for subjectVertex in inputList:
                    e = subjectVertex
                    if inside(e, seg):
                        if not inside(s, seg):
                            outputList.append(computeIntersection(s, e, seg))
                        outputList.append(e)
                    elif inside(s, seg):
                        outputList.append(computeIntersection(s, e, seg))
                    s = e
                if len(outputList) < 3:
                    return [], []
            return outputList

        new_polygon = _clip(polygon, segments)

        return new_polygon

    def cal_inscribed_center_triangle(p1, p2, p3):
        """
        Calculate the inscribed center of triangle
        :param p1, p2, p3: the vertices of the triangle
        :return: the inscribed center of triangle
        """
        v1 = np.array([p2[0] - p3[0], p2[1] - p3[1]])
        v2 = np.array([p3[0] - p1[0], p3[1] - p1[1]])
        v3 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        d3 = np.linalg.norm(v3)
        d = d1 + d2 + d3
        d1 = d1 / d
        d2 = d2 / d
        d3 = d3 / d
        # calculate the inscribed center of triangle
        x = d1 * p1[0] + d2 * p2[0] + d3 * p3[0]
        y = d1 * p1[1] + d2 * p2[1] + d3 * p3[1]
        return [x, y]
    
    if cellOrientation(cell):
        cell = cell[::-1]
    points = cell
    init_points = np.array(points)
    flag = False
    while True:
        # select a point inside the polygon
        center = center_point_inside_polygon(points)
        # if last_center is not None:
        #     flag = (last_center[0] - center[0]) ** 2 + (last_center[1] - center[1]) ** 2 < 1e-9
        # calculate the min distance between the center and the polygon
        d = distance_point_polygon(center, points)
        # calculate the lines composed of vertices
        lines = cal_lines_by_points(points, d, center)
        # Calculate the line parallel to each edge in the polygon with distance d from the center
        # new_points = cal_new_polygon(lines)
        # TODO: Yuanjun
        new_points = cal_new_polygon(points, lines)
        # see if the lines construct a triangle
        # if yes, return the center and radius
        # if not, repeat calculating the line parallel to each edge in the new polygon
        if len(new_points) < 3:
            d = distance_point_polygon(center, init_points)
            return center, d
        elif len(new_points) == 3 or flag:
            p1 = new_points[0]
            p2 = new_points[1]
            p3 = new_points[2]
            center = cal_inscribed_center_triangle(p1, p2, p3)
            d = distance_point_polygon(center, init_points)
            return center, d
        else:
            new_points = np.array(new_points)
            points = new_points
        # repeat calculating the center until the new polygon is a triangle

def computePowerDiagramBruteForce(positions, weights=None, radii=None, clipHull=False, hull=None, dummy=None, vsol=None):
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


def computePowerDiagramByCGAL(positions, weights=None, hull=None):
    if weights is None:
        nonneg_weights = np.zeros(len(positions))
    else:
        nonneg_weights = weights - np.min(weights)

    rt = Regular_triangulation_2()

    v_handles = []
    for pos, w in zip(positions, nonneg_weights):
        v_handle = rt.insert(Weighted_point_2(Point_2(float(pos[0]), float(pos[1])), float(w)))
        v_handles.append(v_handle)

    control_point_set = [
        Weighted_point_2(Point_2(-10, -10), 0),
        Weighted_point_2(Point_2(10, -10), 0),
        Weighted_point_2(Point_2(10, 10), 0),
        Weighted_point_2(Point_2(-10, 10), 0)
    ]

    for cwp in control_point_set:
        rt.insert(cwp)

    cells = []

    # for i, handle in enumerate(v_handles):
    i = 0
    for handle in rt.finite_vertices():
        non_hidden_point = handle.point()
        while i < len(positions) and (non_hidden_point.x() - positions[i, 0])**2 + (non_hidden_point.y()- positions[i, 1])**2 > 1e-10:
            i += 1
            cells.append(Polygon([]))
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
        # ***************************************
        poly_cell = Polygon(cell)
        if hull is not None and not hull.contains(poly_cell):
            poly_cell = hull.intersection(poly_cell)
        cells.append(poly_cell)
        # **************************************
        i += 1

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


class SCPPacking(object):
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
            "optimization": "DivideAndConquer",
            "compaction": "None"
        }
        self.debug = False
        self.logs = {}
        self.preserved_nn_pairs = []
        self.pre_layout_positions = None

        self.extra_params = {}
        # Default params
        force = 85
        self.extra_params['attraction'] = force
        self.extra_params['gravity'] = force
        self.extra_params['utopian'] = True
        self.extra_params['iterations'] = 1000
        self.extra_params['rotation_angle'] = 50
        self.extra_params['top_iters'] = 30
        self.extra_params['cpd_iters'] = 100
        self.extra_params['alpha_min'] = 0.0009

        self.logger = None

        self.intermediate_result = {}

    def __str__(self):
        idf = []
        for name in self.config:
            idf.append(self.config[name][0])
        ret = "3Phase" + "-" + "-".join(idf)
        extra_ret = ''
        for key in self.extra_params:
            extra_ret += '~' + key + '~' + str(self.extra_params[key])
        return ret + extra_ret

    def set_logger(self, logger):
        self.logger = logger

    def run(self):
        initial_pos = self.pre_layout()
        self.pre_layout_positions = initial_pos

        mask = self.build_graph(initial_pos)
        rough_circle_list, rough_links = self.optimize(initial_pos, mask)
        fine_circle_list = self.post_process(rough_circle_list, rough_links)

        positions = [[circle[0], circle[1]] for circle in fine_circle_list]
        radii = [circle[2] for circle in fine_circle_list]

        return positions, radii

    def clustering(self, positions):
        if 'utopian' in self.extra_params and self.extra_params['utopian']:
            clustering_cache = os.path.join(config.data_root, "clustering-utopian", f"{self.data['dataset_name']}-kmeans.pkl")
        else:
            clustering_cache = os.path.join(config.data_root, "clustering", f"{self.data['dataset_name']}-kmeans.pkl")
        if os.path.exists(clustering_cache):
            with open(clustering_cache, "rb") as f:
                cluster_data = pickle.load(f)
                self.cluster = cluster_data['cluster_labels']
        else:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            best_score = 0
            best_clustering_labels = None
            best_nc = -1
            for nc in range(2, 10):
                clustering_model = KMeans(n_clusters=nc, random_state=0)
                clustering_model.fit(positions)
                clustering_labels = clustering_model.predict(positions)
                score = silhouette_score(positions, clustering_labels)
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

    def pre_layout(self):
        cfg = self.config['point-placement']
        if 'utopian' in self.extra_params and self.extra_params['utopian']:
            projection_cache = os.path.join(config.data_root, "projection-utopian", f"{self.data['dataset_name']}-{cfg}.pkl")
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
            cluster_data_path = os.path.join(config.data_root, 'high_dimensional_clustering/' + self.data['dataset_name'] + '-kmeans.pkl')
            with open(cluster_data_path, 'rb') as f:
                cluster_data = pickle.load(f)
            cluster_labels = cluster_data['cluster_labels']
            X = self.data['distance']
            cluster_alpha = 0.5
            for i in np.unique(cluster_labels):
                cluster_indices = np.where(cluster_labels == i)[0]
                X[cluster_indices][:, cluster_indices] *= cluster_alpha

        if cfg == "tSNE":
            model = TSNE(n_components=2, method='exact', metric=metric, perplexity=15, random_state=0, early_exaggeration=6, n_iter=3000)
        elif cfg == "MDS":
            model = MDS(n_components=2, dissimilarity=metric, random_state=0)
        elif cfg == "PCA":
            model = PCA(n_components=2, random_state=0)
        else:
            raise NotImplementedError
        
        if X.shape[1] > 2:
            Y = model.fit_transform(X)
        else: # Two dimensional data
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
        mask = np.zeros((self.N, self.N))
        if cfg == "Delaunay":
            tri = Delaunay(pos)
            for (a, b, c) in tri.simplices:
                mask[a, b] = mask[b, a] = mask[a, c] = mask[c, a] = mask[b, c] = mask[c, b] = 1
        else:
            return None
        return mask

    def optimize(self, initial_pos, mask):
        cfg = self.config['optimization']

        if cfg == "Voronoi" or cfg == "WeightedVoronoi":
            # centroidal voronoi tessellation
            USE_DIVIDE_AND_CONQUER = False
            try:
                USE_DIVIDE_AND_CONQUER = self.extra_params['use_dnc']
            except:
                USE_DIVIDE_AND_CONQUER = False
            margin = 0.05
            ratio = 1e9
            pos = initial_pos * (1 - margin * 2) + margin
            for i in range(initial_pos.shape[0]):
                ratio = min(ratio, margin / self.r[i])
            for i in range(initial_pos.shape[0]):
                for j in range(i + 1, initial_pos.shape[0]):
                    x = pos[j, 0] - pos[i, 0]
                    y = pos[j, 1] - pos[i, 1]
                    d = math.sqrt(x * x + y * y)
                    r = self.r[i] + self.r[j]
                    ratio = min(ratio, d / r)
            radii = self.r * ratio

            def tesellate(indices, p, r):
                iterations = self.extra_params['cpd_iters']

                pos = p[indices].copy()
                rad = r[indices].copy()

                hull = computeConvexHull(pos)

                for _ in range(iterations):
                    if cfg == "WeightedVoronoi":
                        cells = computePowerDiagramByCGAL(pos, rad ** 2, hull=hull)
                        # Relax radii and pos
                        ratio = 1e5
                        for i , cell in enumerate(cells):
                            cell = np.array(cell.exterior.coords[:-1])
                            if 'chebyshev' in self.extra_params and self.extra_params['chebyshev']:
                                centroid, max_rad = cellInscribedCircleRadiusByChebyshev(cell)
                            else:
                                centroid = cellCentroid(cell)
                                max_rad = cellInscribedCircleRadius(cell, centroid)
                            pos[i] = centroid
                            ratio = min(ratio, max_rad / rad[i])
                        rad = rad * ratio

                    else:
                        cells = computePowerDiagramByCGAL(pos, hull=hull)

                        # Relax pos
                        ratio = 1e5
                        for i, cell in enumerate(cells):
                            centroid = cellCentroid(cell)
                            pos[i] = centroid
                            max_rad = cellInscribedCircleRadius(cell, centroid)
                            ratio = min(ratio, max_rad / rad[i])

                h = np.array(hull.exterior.coords[:-1])
                if cfg == "WeightedVoronoi":
                    _, _, pairs = computePowerDiagramBruteForce(positions=pos, weights=rad ** 2, clipHull=True, hull=h)
                else:
                    _, _, pairs = computePowerDiagramBruteForce(positions=pos, clipHull=True, hull=h)

                real_pairs = []
                for (i, j) in pairs:
                    real_pairs.append((indices[i], indices[j]))

                return pos, rad, real_pairs, hull

            scales = []
            partial_results = []
            if USE_DIVIDE_AND_CONQUER:
                for clabel in np.unique(self.cluster):
                    selected_indices = np.where(self.cluster == clabel)[0]
                    partial_res = tesellate(selected_indices, pos, radii)
                    partial_results.append((selected_indices, partial_res))
                    scale = partial_res[1][0] / radii[selected_indices[0]]
                    scales.append(scale)
            else:
                selected_indices = list(range(self.N))
                partial_res = tesellate(selected_indices, pos, radii)
                partial_results.append((selected_indices, partial_res))
                scale = partial_res[1][0] / radii[selected_indices[0]]
                scales.append(scale)

            min_scale = np.min(scale)
            pairs = []
            for scale, (indices, res) in zip(scales, partial_results):
                scaling_factor = min_scale / scale
                cluster_pos, cluster_rad, cluster_pairs, cluster_hull = res
                center = cellCentroid(cluster_hull)
                cluster_pos = (cluster_pos - center) * scaling_factor + center
                cluster_rad = cluster_rad * scaling_factor
                pos[indices] = cluster_pos
                radii[indices] = cluster_rad
                pairs += cluster_pairs

            circle_list = [[pos[i, 0], pos[i, 1], radii[i]] for i in range(self.N)]
            preservation = set(pairs)

        elif cfg == 'DivideAndConquer':
            self.extra_params['center'] = 'hull'
            self.extra_params['update_hull'] = True

            def grad_W(cells, caps):
                grad = []
                for cell, cap in zip(cells, caps):
                    area = cellArea(cell)
                    grad.append(cap - area)
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
                    while n < 20:
                        new_w = w0 + (half ** n) * dk
                        new_cells = computePowerDiagramByCGAL(positions, new_w, hull=hull)                    
                        new_F = F(new_cells, positions, caps, new_w)

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
                scaling_min, scaling_max = 1, 1 # min containt, max not containt
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

            margin = 0.05
            pos = initial_pos * (1 - margin * 2) + margin
            N = self.r.shape[0]
            dis_mat = cdist(pos, pos, metric='euclidean')
            np.fill_diagonal(dis_mat, np.inf)
            r_mat = np.tile(self.r, N).reshape((N, N))
            r_mat += r_mat.T
            ratio = min(np.min(dis_mat / r_mat), margin / np.max(self.r))

            # for i in range(pos.shape[0]):
            #     ratio = min(ratio, margin / self.r[i])
            #     for j in range(i + 1, pos.shape[0]):
            #         x = pos[j, 0] - pos[i, 0]
            #         y = pos[j, 1] - pos[i, 1]
            #         d = math.sqrt(x * x + y * y)
            #         r = self.r[i] + self.r[j]
            #         ratio = min(ratio, d / r)

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

                # ratio = 1e9
                # for i in range(selected_indices.shape[0]):
                #     for j in range(i + 1, selected_indices.shape[0]):
                #         x = pos[selected_indices[j], 0] - pos[selected_indices[i], 0]
                #         y = pos[selected_indices[j], 1] - pos[selected_indices[i], 1]
                #         d = math.sqrt(x * x + y * y)
                #         r = radii[selected_indices[i]] + radii[selected_indices[j]]
                #         ratio = min(ratio, d / r)
                radii[selected_indices] *= ratio

            hull = computeConvexHull(pos)

            top_level_iterations = 10

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
                    max_scaling_ratio, max_scaling_rotation_angle = find_transform(cells[k], cluster_centers[k], pos[selected_indices])
                    pos[selected_indices] = rotateByOrigin(pos[selected_indices] - cc[k], max_scaling_rotation_angle) * max_scaling_ratio + cluster_centers[k]
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

            for k in range(len(np.unique(self.cluster))):
                selected_indices = cluster_indices[k]
                cluster_pos = pos[selected_indices]
                cluster_r = radii[selected_indices]
                param = (k, cluster_pos, cluster_r, cluster_cells[k], self.extra_params, self.r[selected_indices][0])
                params.append(param)

            # Parallel the computing
            # mpi4py version
            # *****************************
            # self.logger.save_as_pkl('params.pkl', params)
            # os.system('C:/"Program Files"/"Microsoft MPI"/Bin/mpiexec.exe -n ' + \
            # # str(MultiNum) + ' "D:/ProgramData/Anaconda3/envs/circle_packing/python.exe" parallel.py ' + self.logger.save_dir + 'params.pkl')
            # str(MultiNum) + ' "python.exe" parallel.py ' + self.logger.save_dir + 'params.pkl')
            pool = Pool(MultiNum)
            processes = [pool.apply_async(single_cell_optimization, param) for param in params]
            res = []
            for i, proc in enumerate(processes):
                res.append(proc.get())

            pairs = []
            min_scale = 1e9
            for it in res:
                min_scale = min(min_scale, it[0])
                selected_indices = cluster_indices[it[2]]
                pos[selected_indices] = it[3]
                radii[selected_indices] = it[4]
                pairs += [(selected_indices[j], selected_indices[k]) for (j, k) in it[1]]

            # ******************************
            radii = self.r * min_scale
            circle_list = [[pos[i, 0], pos[i, 1], radii[i]] for i in range(self.N)]
            preservation = set(pairs)
            
        else:
            raise NotImplementedError

        return circle_list, preservation

    def post_process(self, circle_list, preservation):
        cfg = self.config['compaction']
        initial_pos = np.array([c[:2] for c in circle_list])
        radii = np.array([c[-1] for c in circle_list])
        if cfg == "Box2D":
            config_b2d = {
                "size_mag": 1.0,
                "iterations": self.extra_params['iterations'],
                "attraction_mag": self.extra_params['attraction'],
                "density": 0,
                "gravity": self.extra_params['gravity'],
                "bullet": False,
                "time_step": 0.005,
            }

            alpha = 1
            alpha_min = self.extra_params['alpha_min']
            alpha_decay = (1 - (alpha_min ** (1 / config_b2d["iterations"])))
            alpha_target = 0

            if self.config["optimization"] == "DivideAndConquer":
                attractions = [(i, j, self.data['similarity'][i, j]) for (i, j) in preservation if self.cluster[i] == self.cluster[j]]
            elif self.config["optimization"] == "WeightedVoronoi":
                attractions = [(i, j, self.data['similarity'][i, j]) for (i, j) in preservation]

            pre_pos = initial_pos.copy()
            rad = radii.copy()

            config_b2d["size_mag"] = 1 / np.min(rad)
            config_b2d["gravity"] *= config_b2d["size_mag"]
            config_b2d["attraction_mag"] *= config_b2d["size_mag"]
            optimizer = Box2DSimulator(positions=pre_pos * config_b2d["size_mag"],
                                       radii=rad * config_b2d["size_mag"], size_mag=config_b2d["size_mag"], attractions=attractions,
                                       attraction_magnitude=config_b2d["attraction_mag"], density=config_b2d["density"],
                                       gravity=config_b2d["gravity"], bullet=config_b2d["bullet"],
                                       time_step=config_b2d["time_step"])
            
            for _ in range(config_b2d["iterations"]):
                optimizer.clear_velocities()
                optimizer.apply_attractions_and_gravity()
                optimizer.step()
                alpha += alpha_decay * (alpha_target-alpha)
                optimizer.attraction_magnitude = alpha * config_b2d["attraction_mag"]
                optimizer.gravity = alpha * config_b2d["gravity"]
                if alpha <= alpha_min:
                    break

            pos = np.array(optimizer.get_positions())
            rad = np.array(optimizer.get_radii())

            resolution = max(np.max(pos[:, 0] + rad) - np.min(pos[:, 0] - rad),
                             np.max(pos[:, 1] + rad) - np.min(pos[:, 1] - rad))
            pos = (pos - (pos - rad.reshape(-1, 1)).min(axis=0)) / resolution
            rad /= resolution
            circle_list = [[pos[i, 0], pos[i, 1], rad[i]] for i in range(len(pos))]

        elif cfg == "None":
            pass
        else:
            raise NotImplementedError

        return circle_list
