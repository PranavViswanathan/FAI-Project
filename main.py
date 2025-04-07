'''
Extends Gym’s CarRacing environment to allow for customized track parameters and an optional fork feature (branching paths), with added debug/log messages and a method to retrieve the car’s current state.
'''
import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

class CustomCarRacing(CarRacing):

    def __init__(self, track_complexity=1.0, track_width=1.0, num_forks=0):
        super().__init__(render_mode="human")
        self.track_complexity = track_complexity
        self.track_width = track_width
        self.num_forks = num_forks

        print("[INFO] CustomCarRacing initialized with:")
        print(f" - Track Complexity: {self.track_complexity}")
        print(f" - Track Width: {self.track_width}")
        print(f" - Number of Forks: {self.num_forks}")

    def _create_track(self):
        if hasattr(self, 'track_created') and self.track_created:
            return
        self.track_created = True

        print("[DEBUG] _create_track() running...")
        success = False
        attempt = 0

        while not success:
            attempt += 1
            print(f"[DEBUG] Attempt {attempt}: Trying to generate a valid track...")
            self.road = []
            self.road_poly = []
            self.track = []
            self.TRACK_DETAIL_STEP = int(21 * self.track_complexity)
            self.TRACK_TURN_RATE = 0.31 * self.track_complexity
            self.TRACK_WIDTH = 40 * self.track_width

            success = super()._create_track()

            if not success:
                print(f"[WARNING] Track generation failed, retrying... (Attempt {attempt})")
            if attempt > 5:
                raise RuntimeError("[ERROR] Track generation failed after 5 attempts!")

        print(f"[INFO] Track generation successful after {attempt} attempts!")

    def _add_forks(self):
        """Optional: Add forks (alternative paths) in the track."""
        print("[INFO] Adding forks to the track...")
        fork_chance = 0.1
        fork_offset = 10
        new_segments = []
        for i in range(len(self.track)):
            if np.random.rand() < fork_chance and len(new_segments) < self.num_forks:
                fork_point = self.track[i]
                fork_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
                new_x = fork_point[0] + fork_offset * np.cos(fork_angle)
                new_y = fork_point[1] + fork_offset * np.sin(fork_angle)
                new_segments.append((new_x, new_y))
        if new_segments:
            print(f"[INFO] Added {len(new_segments)} fork points.")
        self.track.extend(new_segments)

    def get_state(self):
        if not self.track:
            return np.array([0.0, 0.0, 0.0])

        car_x, car_y = self.car.hull.position
        nearest_point = min(self.track, key=lambda p: np.linalg.norm([p[0] - car_x, p[1] - car_y]))
        dx, dy = nearest_point[0] - car_x, nearest_point[1] - car_y
        lateral_offset = np.clip(np.linalg.norm([dx, dy]), -1.0, 1.0)

        car_angle = self.car.hull.angle
        heading_angle = car_angle  # This can be improved with track tangent approximation

        speed = self.car.hull.linearVelocity.length

        state = np.array([lateral_offset, heading_angle, speed])
        print(f"[DEBUG] State: offset={lateral_offset:.3f}, angle={heading_angle:.3f}, speed={speed:.2f}")
        return state

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

# Custom environment registration (optional if you're not using gym.make("CustomCarRacing-v0"))
gym.envs.registration.register(
    id="CustomCarRacing-v0",
    entry_point="main:CustomCarRacing",
    max_episode_steps=5,
)
