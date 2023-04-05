# simulate circles using box2d
import numpy as np
from Box2D import *
from utils.ApplyForce.ApplyForce import ApplyForce


class Box2DSimulator(object):
    def __init__(self, positions, radii, size_mag, attractions, attraction_magnitude, iterations = 0, density=1.0, gravity=10, bullet=False,
                 time_step=0.005,
                 velocity_iterations=6, position_iterations=2):
        self.positions = positions
        self.radii = radii
        self.size_mag = size_mag
        self.attractions = attractions
        self.attraction_magnitude = attraction_magnitude
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
        self.ForceApplicator.set_rad(self.radii)
        self.ForceApplicator.set_attraction_pairs(self.attractions)
 
    def init_bodies(self):
        for i in range(len(self.positions)):
            # convert to float
            position = (float(self.positions[i][0]), float(self.positions[i][1]))
            body = self.world.CreateDynamicBody(position=position, angle=0, linearDamping=0, angularDamping=0, bullet = self.bullet)
            body.CreateCircleFixture(radius=self.radii[i], density=self.density, friction=0)
            self.bodies.append(body)

    def apply_attractions_and_gravity(self):
        self.ForceApplicator.set_pos([body.position for body in self.bodies])
        attraction_forces = self.ForceApplicator.get_attraction(self.attraction_magnitude)
        gravities = self.ForceApplicator.get_gravity(self.gravity, self.size_mag)
        for i in range(len(self.bodies)):
            self.bodies[i].ApplyForceToCenter(attraction_forces[i], True)
            self.bodies[i].ApplyForceToCenter(gravities[i], True)

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

    def set_attraction_magnitude(self, attraction_magnitude):
        self.attraction_magnitude = attraction_magnitude
