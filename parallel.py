import pickle
import sys
import numpy as np
from mpi4py import MPI
import math

from backend.utils.Voronoi import Voronoi
from CGAL.CGAL_Kernel import *
from CGAL.CGAL_Triangulation_2 import *
from shapely.geometry import Polygon


def cross_product(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]


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
        # h = np.clip(h, 0, 1)
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
            points = np.array(new_points)
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

def single_cell_optimization(k, cluster_pos, cluster_r, cluster_cell, extra_params, r0, filename):
    cluster_hull = Polygon(cluster_cell)
    iterations = 30
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
    with open(filename, 'wb') as f:
        pickle.dump((r_scale, pairs, k, cluster_pos, cluster_r), f)
    return r_scale, pairs, k, cluster_pos, cluster_r

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    filename = sys.argv[1]
    with open(filename, 'rb') as f:
        params = pickle.load(f)
else:
    params = None

params = comm.bcast(params, root=0)

single_cell_optimization(*params[rank])
