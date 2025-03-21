import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

class CustomCarRacing(CarRacing):
    def __init__(self, track_complexity=1.0, track_width=1.0, num_forks=0):
        super().__init__(render_mode="rgb_array")
        self.track_complexity = track_complexity  # Higher = more turns
        self.track_width = track_width  # Higher = wider track
        self.num_forks = num_forks  # Number of alternative paths

        print("[INFO] CustomCarRacing initialized with:")
        print(f" - Track Complexity: {self.track_complexity}")
        print(f" - Track Width: {self.track_width}")
        print(f" - Number of Forks: {self.num_forks}")

    def _create_track(self):
        if hasattr(self, "track_generated") and self.track_generated:
            print("[WARNING] _create_track() was called again, but track is already generated!")
            return 
        self.track_generated = True  

        print("[DEBUG] _create_track() is running for the first time...")

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
        """Introduce forks (alternative paths) in the track."""
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

        self.track.extend(new_segments) # Forking in thr road
    
    # def reset(self, seed=None, options=None):
    #     """Ensure reset() is not being called in a loop."""
    #     print("[INFO] CustomCarRacing: Reset started...")

    #     # Ensure _create_track() only runs once per reset
    #     if not hasattr(self, "track_created"):
    #         self.track_created = False

    #     if not self.track_created:
    #         print("[DEBUG] Calling _create_track() in reset()...")
    #         obs, info = super().reset(seed=seed, options=options)
    #         self.track_created = True  # Prevent infinite resets
    #         print("[INFO] CustomCarRacing: Reset completed successfully!")
    #         return obs, info
    #     else:
    #         print("[WARNING] reset() was called again unexpectedly.")
    #         return None, None
    
    def reset(self, seed=None, options=None):
        """Ensure reset() does not regenerate the track if it already exists."""
        print("[INFO] CustomCarRacing: Reset started...")

        if not hasattr(self, "track_created"):
            self.track_created = False

        if not self.track_created:
            print("[DEBUG] First-time reset, generating track...")
            self.track_created = True
            obs, info = super().reset(seed=seed, options=options)
        else:
            print("[WARNING] reset() called again, skipping track generation.")
            obs = np.zeros((96, 96, 3))  
            info = {}

        print("[INFO] CustomCarRacing: Reset completed successfully!")
        return obs, info



# Custom environment registration
gym.envs.registration.register(
    id="CustomCarRacing-v0",
    entry_point="custom_car_racing:CustomCarRacing",
    max_episode_steps=1000,
)
