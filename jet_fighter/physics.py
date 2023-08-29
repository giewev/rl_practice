import numpy as np
import pygame

class PhysicsObject:
    def __init__(self, bounding_box, max_linear_velocity=20, max_angular_velocity=0.3):
        self.position = np.zeros(2)
        self.linear_velocity = np.zeros(2)
        self.heading = 0
        self.angular_velocity = 0
        self.rotation_damping = 1.0
        self.linear_damping = 1.0
        self.bounding_box = np.array(bounding_box)
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

    def apply_acceleration(self, linear_acceleration, angular_acceleration, reference_frame='absolute'):
        if linear_acceleration != 0 and reference_frame == 'relative':
            rotation_matrix = np.array([
                [np.cos(self.heading), -np.sin(self.heading)],
                [np.sin(self.heading), np.cos(self.heading)]
            ])
            linear_acceleration = rotation_matrix.dot(linear_acceleration)
        self.linear_velocity += linear_acceleration
        self.angular_velocity += angular_acceleration
        self.limit_velocity()

    def update(self):
        self.linear_velocity *= self.linear_damping
        self.angular_velocity *= self.rotation_damping
        # print(self.linear_velocity)
        self.position += self.linear_velocity
        self.heading += self.angular_velocity
        self.wrap_around()

    def wrap_around(self):
        self.position %= self.bounding_box
        self.heading %= 2 * np.pi

    def relative_position_of(self, other_object):
        relative_position = other_object.position - self.position
        rotation_matrix = np.array([
            [np.cos(-self.heading), -np.sin(-self.heading)],
            [np.sin(-self.heading), np.cos(-self.heading)]
        ])
        return rotation_matrix.dot(relative_position)
    
    def relative_bearing_of(self, other_object):
        relative_position = self.relative_position_of(other_object)
        angle = np.arctan2(relative_position[1], relative_position[0])
        return (angle + 2 * np.pi) % (2 * np.pi)

    def wrapped_distance(self, other_object):
        delta = np.abs(self.position - other_object.position)
        wrapped_delta = np.minimum(delta, self.bounding_box - delta)
        return np.linalg.norm(wrapped_delta)

    def reset(self):
        self.position.fill(0)
        self.linear_velocity.fill(0)
        self.heading = 0
        self.angular_velocity = 0

    def randomize_position(self):
        self.position = np.random.uniform(0, self.bounding_box)

    def randomize_velocity(self):
        direction = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, self.max_linear_velocity)
        self.linear_velocity = np.array([np.cos(direction), np.sin(direction)]) * magnitude
        # print(self.linear_velocity)

    def randomize_heading(self):
        self.heading = np.random.uniform(0, 2 * np.pi)

    def randomize_angular_velocity(self):
        self.angular_velocity = np.random.uniform(-self.max_angular_velocity, self.max_angular_velocity)

    def limit_velocity(self):
        velocity_magnitude = np.linalg.norm(self.linear_velocity)
        if velocity_magnitude > self.max_linear_velocity:
            self.linear_velocity *= self.max_linear_velocity / velocity_magnitude
        self.angular_velocity = np.clip(self.angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

    def render(self, screen, color):
        target_radius = 10
        pygame.draw.circle(
            screen,
            color,
            (
                int(self.position[0] * self.bounding_box[0] / self.bounding_box[0]),
                int(self.position[1] * self.bounding_box[1] / self.bounding_box[1]),
            ),
            target_radius,
        )

class Jet(PhysicsObject):
    def __init__(self, bounding_box):
        super(Jet, self).__init__(bounding_box, 30, np.pi/8)
        self.linear_thrust = 1
        self.rotational_thrust = np.pi/40
        self.rotation_damping = 0.99
        self.linear_damping = 0.99

    def thrust(self):
        self.apply_acceleration([self.linear_thrust, 0], 0, reference_frame='relative')

    def spin_cw(self):
        self.apply_acceleration(0, -self.rotational_thrust)

    def spin_ccw(self):
        self.apply_acceleration(0, self.rotational_thrust)
    
    def render(self, screen, color):
        jet_size = 10
        mini_box = self.bounding_box / 2
        rads = 140 * 2 * np.pi / 360

        for i in range(0, 2):
            for j in range(0, 2):
                shifted_pos = self.position/2 + (mini_box * np.array([i, j]))
                # print(self.position, shifted_pos)
                jet_points = [
                    (
                        shifted_pos[0] + jet_size * np.cos(self.heading),
                        shifted_pos[1] + jet_size * np.sin(self.heading),
                    ),
                    (
                        shifted_pos[0] + jet_size * np.cos(self.heading + rads),
                        shifted_pos[1] + jet_size * np.sin(self.heading + rads),
                    ),
                    (
                        shifted_pos[0] + jet_size * np.cos(self.heading - rads),
                        shifted_pos[1] + jet_size * np.sin(self.heading - rads),
                    ),
                ]
                pygame.draw.polygon(screen, color, jet_points)