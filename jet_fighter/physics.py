import numpy as np

class PhysicsObject:
    def __init__(self, bounding_box, max_linear_velocity=20, max_angular_velocity=0.3):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.angular_position = 0
        self.angular_velocity = 0
        self.rotation_damping = 0.99
        self.linear_damping = 0.99
        self.bounding_box = np.array(bounding_box)
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

    def apply_acceleration(self, linear_acceleration, angular_acceleration, reference_frame='absolute'):
        if reference_frame == 'relative':
            rotation_matrix = np.array([
                [np.cos(self.angular_position), -np.sin(self.angular_position)],
                [np.sin(self.angular_position), np.cos(self.angular_position)]
            ])
            linear_acceleration = rotation_matrix.dot(linear_acceleration)
        self.velocity += linear_acceleration
        self.angular_velocity += angular_acceleration
        self.limit_velocity()

    def update(self, dt):
        self.position += self.velocity * dt
        self.angular_position += self.angular_velocity * dt
        self.wrap_around()

    def wrap_around(self):
        self.position %= self.bounding_box

    def relative_position_of(self, other_object):
        relative_position = other_object.position - self.position
        rotation_matrix = np.array([
            [np.cos(-self.angular_position), -np.sin(-self.angular_position)],
            [np.sin(-self.angular_position), np.cos(-self.angular_position)]
        ])
        return rotation_matrix.dot(relative_position)

    def reset(self):
        self.position.fill(0)
        self.velocity.fill(0)
        self.angular_position = 0
        self.angular_velocity = 0

    def randomize_position(self):
        self.position = np.random.uniform(0, self.bounding_box)

    def randomize_velocity(self):
        self.velocity = np.random.uniform(-self.max_linear_velocity, self.max_linear_velocity, size=2)

    def randomize_angular_position(self):
        self.angular_position = np.random.uniform(0, 2 * np.pi)

    def randomize_angular_velocity(self):
        self.angular_velocity = np.random.uniform(-self.max_angular_velocity, self.max_angular_velocity)

    def limit_velocity(self):
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > self.max_linear_velocity:
            self.velocity *= self.max_linear_velocity / velocity_magnitude
        self.angular_velocity = np.clip(self.angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

class Jet(PhysicsObject):
    def __init__(self):
        super(Jet)
        self.max_linear_velocity = 30
        self.max_angular_velocity = 10
        self.max_angular_velocity

    def thrust(self, amount):
        forward_acceleration = np.array([np.cos(self.angular_position), np.sin(self.angular_position)]) * amount
        self.apply_acceleration(forward_acceleration, 0)

    def spin_cw(self, amount):
        self.apply_acceleration(0, -amount)

    def spin_ccw(self, amount):
        self.apply_acceleration(0, amount)