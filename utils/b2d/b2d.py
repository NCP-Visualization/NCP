# simulate circles using box2d
import numpy as np
from Box2D import *
from utils.ApplyForce.ApplyForce import ApplyForce
# from scipy.spatial import ConvexHull
# import alphashape
import math
from CGAL.CGAL_Kernel import *
from CGAL.CGAL_Triangulation_2 import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import shapely
import time, os
from multiprocessing.pool import ThreadPool
from scipy.spatial import Delaunay
from utils.EPD import EPD


def cellArea(cell):
    area = 0
    for k in range(len(cell)):
        p1 = cell[k]
        p2 = cell[(k + 1) % len(cell)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    area /= 2
    return area


def get_concave_hull(pos, r=None, last_a=None):
    if last_a is None:
        a = alphashape.optimizealpha(pos, max_iterations=50, lower=0.0, upper=1.0)
    else:
        a = alphashape.optimizealpha(pos, max_iterations=5, lower=last_a*0.9, upper=last_a*1.1)
    concave_hull = alphashape.alphashape(pos, a)
    coords = np.array(concave_hull.exterior.coords[:-1])
    hull_indices = []
    for coord in coords:
        for j, p in enumerate(pos):
            if np.sum((p - coord) ** 2) < 1e-12:
                hull_indices.append(j)
                break
    return concave_hull, hull_indices, a


def alphasimplices(points):
    """
    Returns an iterator of simplices and their circumradii of the given set of
    points.

    Args:
      points: An `N`x`M` array of points.

    Yields:
      A simplex, and its circumradius as a tuple.
    """
    coords = np.asarray(points)
    tri = Delaunay(coords)

    epd_instance = EPD.EPD()
    epd_instance.set_cluster_pos(coords)
    for simplex in tri.simplices:
        yield simplex, epd_instance.circumradius(simplex)


def alphashape(points, alpha):
    alpsps = list(alphasimplices(points))
    # t0 = time.time()
    esol = EPD.EPD()
    result = esol.concave_hull_indices(alpsps, alpha, len(points))
    if len(result) > 0 and cellArea(points[result]) > 0:
        result.reverse()
    # t1 = time.time()
    # print("ttt1", t1 - t0)
    return result


def fast_concave_hull_approx(points, upper):
    test_alpha = upper
    count = 0
    while True:
        # Bisect the current bounds
        test_alpha *= .5
        count += 1
        # t1=time.time()
        polygon = alphashape(points, test_alpha)
        # t2=time.time()
        # print(t2-t1, points.shape)
        # if isinstance(polygon, shapely.geometry.polygon.Polygon):
        if len(polygon) > 0:
            # print("check", count, test_alpha)
            return polygon


def get_concave_hull_approx(pos, rad):
    if len(pos) < 3:
        return []

    concave_hull = fast_concave_hull_approx(pos, 1.0 / np.min(rad))
    # print(concave_hull)
    return concave_hull

    # coords = np.array(concave_hull.exterior.coords[:-1])
    #
    # hull_indices = []
    # for coord in coords:
    #     for j, p in enumerate(pos):
    #         if np.sum((p - coord) ** 2) <= rad[j] ** 2 + 1e-12:
    #             hull_indices.append(j)
    #             break
    # return hull_indices


def get_concave(pos,rad,groups):
    pos = np.array(pos)
    # chain = get_concave_hull_approx(pos, rad)
    # hull_indices = chain
    # part_hull_indices = []
    #
    # for indices in groups:
    #     tmp = get_concave_hull_approx(pos[indices], rad[indices])
    #     chain_part = indices[tmp]
    #     part_hull_indices.append(chain_part)
    #
    # n_hull = []
    # all_hull_indices = []
    # n_hull.append(len(hull_indices))
    # for id in hull_indices:
    #     all_hull_indices.append(id)
    # for indices in part_hull_indices:
    #     n_hull.append(len(indices))
    #     for id1 in indices:
    #         all_hull_indices.append(id1)

    # MultiNum = len(groups)
    # pool = ThreadPool(MultiNum+1)
    #
    # part_hull_indices = []
    # processes = [pool.apply_async(get_concave_hull_approx, param) for param in [(pos, rad)] + [(pos[indices], rad[indices]) for indices in groups]]
    # for i, proc in enumerate(processes):
    #     part_hull_indices.append(proc.get())
    #
    # n_hull = []
    # all_hull_indices = []
    # for k, indices in enumerate(part_hull_indices):
    #     n_hull.append(len(indices))
    #     if k != 0:
    #         indices = groups[k-1][indices]
    #     for id1 in indices:
    #         all_hull_indices.append(id1)

    MultiNum = len(groups)
    pool = ThreadPool(MultiNum)

    part_hull_indices = []
    processes = [pool.apply_async(get_concave_hull_approx, param) for param in [(pos[indices], rad[indices]) for indices in groups]]
    for i, proc in enumerate(processes):
        part_hull_indices.append(proc.get())

    n_hull = []
    all_hull_indices = []
    for k, indices in enumerate(part_hull_indices):
        n_hull.append(len(indices))
        indices = groups[k][indices]
        for id1 in indices:
            all_hull_indices.append(id1)

    # print(n_hull, all_hull_indices)

    return n_hull, all_hull_indices


class Box2DSimulator(object):
    def __init__(self, positions, radii, cluster, size_mag, attractions, attraction_magnitude, convexity_magnitude, iterations=0, density=1.0, gravity=10, bullet=False,
                 time_step=0.005,
                 velocity_iterations=6, position_iterations=2):
        self.positions = positions
        self.radii = radii
        self.cluster = cluster
        self.size_mag = size_mag
        self.attractions = attractions
        self.attraction_magnitude = attraction_magnitude
        self.convexity_magnitude = convexity_magnitude
        self.iterations = iterations
        self.density = density
        self.gravity = gravity
        self.bullet = bullet
        self.time_step = time_step
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self.fixtures = []
        self.colors = []
        self.init_bodies()
        self.applied_gravities = None
        self.applied_attractions = None
        self.attraction_potential = None
        self.repulsion_potential = None
        self.gravity_potential = None
        self.ForceApplicator = ApplyForce()
        self.ForceApplicator.set_n(len(self.bodies))
        # self.ForceApplicator.set_time_step(time_step)

        cluster_label_mapping = {}
        for i, cluster_label in enumerate(np.unique(self.cluster)):
            cluster_label_mapping[cluster_label] = i
        new_cluster = np.array([cluster_label_mapping[c] for c in self.cluster])
        self.groups = []
        for i in range(len(cluster_label_mapping)):
            self.groups.append(np.where(new_cluster == i)[0])

        # self.ForceApplicator.set_cluster(new_cluster, len(cluster_label_mapping))
        self.ForceApplicator.set_rad(self.radii)
        self.ForceApplicator.set_attraction_pairs(self.attractions)

        # Monitoring
        self.hull_indices = []
        self.part_hull_indices = []
        # self.forces = {}
 
    def init_bodies(self):
        for i in range(len(self.positions)):
            # convert to float
            position = (float(self.positions[i][0]), float(self.positions[i][1]))
            body = self.world.CreateDynamicBody(position=position, angle=0, linearDamping=0, angularDamping=0, bullet=self.bullet)
            body.CreateCircleFixture(radius=float(self.radii[i]), density=self.density, friction=0)
            self.bodies.append(body)

    def apply_attractions_and_gravity(self, conv=False):
        pos = np.array([body.position for body in self.bodies])
        self.ForceApplicator.set_pos(pos)
        # self.ForceApplicator.set_pos([body.position for body in self.bodies])
        attraction_forces = np.array(self.ForceApplicator.get_attraction(self.attraction_magnitude))
        gravities = np.array(self.ForceApplicator.get_gravity(self.gravity, self.size_mag))

        if conv and self.convexity_magnitude > 0:
            # t1 = time.time()
            # chain = get_concave_hull_approx(pos, self.radii)
            # self.hull_indices = chain
            # print(chain)

            self.part_hull_indices = []

            for indices in self.groups:
                tmp = get_concave_hull_approx(pos[indices], self.radii[indices])
                chain_part = indices[tmp]
                self.part_hull_indices.append(chain_part)
            # print(self.part_hull_indices)

            # parallel compute
            # Multinum = len(self.groups)
            # pool = Pool(Multinum)
            #
            # self.part_hull_indices = pool.starmap(get_concave_hull_approx, [(pos[indices], self.radii[indices]) for indices in self.groups])
            # pool.close()
            # pool.join()

            # processes = [pool.apply_async(get_concave_hull_approx, args=(
            #     pos[indices], self.radii[indices])) for indices in self.groups]
            #
            # for i, proc in enumerate(processes):
            #     tmp = proc.get()
            #     indices = self.groups[i]
            #     chain_part = indices[tmp]
            #     self.part_hull_indices.append(chain_part)
            # t2 = time.time()
            # print("concave", t2-t1)

        # convexities = np.array(self.ForceApplicator.get_convexity(self.hull_indices, self.convexity_magnitude))
        convexities = np.zeros(pos.shape)
        for chain_part in self.part_hull_indices:
            convexities += np.array(self.ForceApplicator.get_convexity(chain_part, self.convexity_magnitude))

        for i in range(len(self.bodies)):
            self.bodies[i].ApplyForceToCenter(attraction_forces[i], True)
            self.bodies[i].ApplyForceToCenter(gravities[i], True)
            self.bodies[i].ApplyForceToCenter(convexities[i], True)

    def clear_velocities(self):
        for body in self.bodies:
            body.linearVelocity = (0, 0)
            body.angularVelocity = 0

    def step(self):
        self.world.Step(self.time_step, self.velocity_iterations, self.position_iterations)

    def get_positions(self):
        return [body.position for body in self.bodies]

    def get_radii(self):
        return [body.fixtures[0].shape.radius for body in self.bodies]

    def get_concave_hull_indices(self):
        return self.hull_indices

    def get_cluster_concave_hull_indices(self):
        return self.part_hull_indices

    def get_forces(self):
        return self.forces

    def set_attraction_magnitude(self, attraction_magnitude):
        self.attraction_magnitude = attraction_magnitude
