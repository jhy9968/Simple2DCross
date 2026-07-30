from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces
from gymnasium.error import DependencyNotInstalled


class Simple2DCross(Env):
    """A simple 2D navigation task for a single controlled pedestrian."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, num_pedestrians: int = 3, render_mode: Optional[str] = None):
        if num_pedestrians < 1:
            raise ValueError("num_pedestrians must be at least 1")

        self.num_pedestrians = num_pedestrians + 1
        self.render_mode = render_mode
        self.max_steps = 200
        self.dt = 0.1
        self.max_speed = 1.0
        self.goal_radius = 0.2
        self.collision_radius = 0.15
        self.world_size = 4.0
        self.agent_box = [2.0, 1.0] # Dimension of agent start and goal box
        self.ped_box = [1.0, 3.0]   # Dimension of pedestrian start and goal box
        self.repulsion_strength = 2.0
        self.repulsion_radius = 1.0

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6 + 4 * max(0, num_pedestrians - 1),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._pedestrians: List[Dict[str, np.ndarray]] = []
        self._rng = np.random.default_rng()

    def _initialize_pedestrians(self) -> None:
        N = self.num_pedestrians
        ws = self.world_size

        # 1. Pre-allocate arrays for starts and goals
        starts = np.empty((N, 2), dtype=np.float32)
        goals = np.empty((N, 2), dtype=np.float32)

        # 2. Agent (Index 0): Bottom to Top
        # rng.uniform can take arrays for bounds [low_x, low_y], [high_x, high_y]
        starts[0] = self._rng.uniform(
            [-self.agent_box[0] / 2, -ws],
            [self.agent_box[0] / 2, -ws + self.agent_box[1]],
        )
        goals[0] = self._rng.uniform(
            [-self.agent_box[0] / 2, ws - self.agent_box[1]],
            [self.agent_box[0] / 2, ws],
        )

        # 3. Obstacles (Index 1+): Left to Right
        if N > 1:
            starts[1:] = self._rng.uniform(
                [-ws, -self.ped_box[1] / 2],
                [-ws + self.ped_box[0], self.ped_box[1] / 2],
                size=(N - 1, 2),
            )
            goals[1:] = self._rng.uniform(
                [ws - self.ped_box[0], -self.ped_box[1] / 2],
                [ws, self.ped_box[1] / 2],
                size=(N - 1, 2),
            )

        # 4. Enforce minimum distance constraint across all pedestrians at once
        too_close = np.linalg.norm(goals - starts, axis=1) < 0.5
        goals[too_close] = starts[too_close] + np.array([1.0, 0.0], dtype=np.float32)

        # 5. Clip all goals at once
        goals = np.clip(goals, -ws, ws)

        # 6. Build the final list of dictionaries
        self._pedestrians = [
            {"pos": starts[i], "vel": np.zeros(2, dtype=np.float32), "goal": goals[i]}
            for i in range(N)
        ]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (vector / norm).astype(np.float32)

    def _social_force_step(self, ped_idx: int) -> None:
        ped = self._pedestrians[ped_idx]
        direction = ped["goal"] - ped["pos"]
        distance_to_goal = float(np.linalg.norm(direction))
        if distance_to_goal < self.goal_radius:
            ped["vel"] = np.zeros(2, dtype=np.float32)
            return
        desired = self._normalize(direction) * self.max_speed

        repulsion = np.zeros(2, dtype=np.float32)
        for other_idx, other in enumerate(self._pedestrians):
            if other_idx == ped_idx:
                continue
            delta = ped["pos"] - other["pos"]
            distance = float(np.linalg.norm(delta))
            if distance < self.repulsion_radius and distance > 1e-8:
                influence = (
                    1.0 - distance / self.repulsion_radius
                ) * self.repulsion_strength
                repulsion += (delta / distance) * influence

        combined = desired + repulsion
        combined = np.clip(combined, -self.max_speed, self.max_speed)
        ped["vel"] = combined.astype(np.float32)
        ped["pos"] = np.clip(
            ped["pos"] + ped["vel"] * self.dt, -self.world_size, self.world_size
        )

    def auto_action(
        self, repulsion_radius: float = 1.0, repulsion_strength: float = 2.0
    ) -> np.ndarray[np.float32, np.float32]:
        agent = self._pedestrians[0]
        direction = agent["goal"] - agent["pos"]
        distance_to_goal = float(np.linalg.norm(direction))
        if distance_to_goal < self.goal_radius:
            agent["vel"] = np.zeros(2, dtype=np.float32)
            return np.array([0, 0], dtype=np.float32)
        desired = self._normalize(direction) * self.max_speed

        repulsion = np.zeros(2, dtype=np.float32)
        for other_idx, other in enumerate(self._pedestrians):
            if other_idx == 0:
                continue
            delta = agent["pos"] - other["pos"]
            distance = float(np.linalg.norm(delta))
            if distance < repulsion_radius and distance > 1e-8:
                influence = (1.0 - distance / repulsion_radius) * repulsion_strength
                repulsion += (delta / distance) * influence

        combined = desired + repulsion
        combined = np.clip(combined, -self.max_speed, self.max_speed)
        return combined.astype(np.float32)

    def _agent_step(self, action: np.ndarray) -> None:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        agent = self._pedestrians[0]
        agent["vel"] = action
        agent["pos"] = np.clip(
            agent["pos"] + agent["vel"] * self.dt, -self.world_size, self.world_size
        )

    def _observe(self) -> np.ndarray:
        agent = self._pedestrians[0]
        observation = [
            agent["pos"][0],
            agent["pos"][1],
            agent["vel"][0],
            agent["vel"][1],
        ]
        for ped in self._pedestrians[1:]:
            observation.extend(
                [ped["pos"][0], ped["pos"][1], ped["vel"][0], ped["vel"][1]]
            )
        observation.extend(
            [agent["goal"][0] - agent["pos"][0], agent["goal"][1] - agent["pos"][1]]
        )
        return np.asarray(observation, dtype=np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._initialize_pedestrians()
        observation = self._observe()
        return observation, {"step": 0}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise ValueError("action must be a 2D continuous vector")

        self._agent_step(action)
        for idx in range(1, self.num_pedestrians):
            self._social_force_step(idx)

        self._step_count += 1
        agent = self._pedestrians[0]
        distance_to_goal = float(np.linalg.norm(agent["pos"] - agent["goal"]))

        terminated = distance_to_goal <= self.goal_radius
        truncated = self._step_count >= self.max_steps

        reward = -0.1 * distance_to_goal
        if terminated:
            reward += 10.0
        if self._collision_detected():
            reward -= 5.0
            terminated = True

        observation = self._observe()
        info = {"distance_to_goal": distance_to_goal, "step": self._step_count}
        return observation, reward, terminated, truncated, info

    def _collision_detected(self) -> bool:
        agent = self._pedestrians[0]
        for other in self._pedestrians[1:]:
            if (
                float(np.linalg.norm(agent["pos"] - other["pos"]))
                < self.collision_radius * 2
            ):
                return True
        return False

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        # 1. Initialize Pygame window on the first render call
        if not hasattr(self, "window") or self.window is None:
            pygame.init()
            pygame.display.init()
            self.window_size = 600

            if self.render_mode == "human":
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                pygame.display.set_caption("Simple2DCross Env")
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))

            self.clock = pygame.time.Clock()

        # 2. Setup the Canvas (White background)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Helper function to convert env coordinates (-world_size to +world_size)
        # to Pygame pixel coordinates (0 to window_size). Pygame's Y-axis is inverted.
        scale = self.window_size / (2 * self.world_size)

        def to_pixels(pos):
            x, y = pos
            return (
                int((x + self.world_size) * scale),
                int((self.world_size - y) * scale),
            )

        # 3. Draw Grid Lines (optional, for visual reference)
        for x in range(int(-self.world_size), int(self.world_size) + 1):
            pygame.draw.line(
                canvas,
                (220, 220, 220),
                to_pixels((x, -self.world_size)),
                to_pixels((x, self.world_size)),
            )
            pygame.draw.line(
                canvas,
                (220, 220, 220),
                to_pixels((-self.world_size, x)),
                to_pixels((self.world_size, x)),
            )

        radius_px = int(getattr(self, "collision_radius", 0.15) * scale)
        agent = self._pedestrians[0]

        # Draw start and goal box of agent
        agent_box_width = int(self.agent_box[0] * scale)
        agent_box_height = int(self.agent_box[1] * scale)
        agent_start_top_left = to_pixels(
            (-self.agent_box[0] / 2, -self.world_size + self.agent_box[1])
        )
        agent_goal_top_left = to_pixels((-self.agent_box[0] / 2, self.world_size))
        agent_box_surface = pygame.Surface(
            (agent_box_width, agent_box_height), pygame.SRCALPHA
        )
        agent_box_surface.fill((180, 255, 180, 100))

        # Draw start and goal box of pedestrians
        ped_box_width = int(self.ped_box[0] * scale)
        ped_box_height = int(self.ped_box[1] * scale)
        ped_start_top_left = to_pixels((-self.world_size, self.ped_box[1] / 2))
        ped_goal_top_left = to_pixels(
            (self.world_size - self.ped_box[0], self.ped_box[1] / 2)
        )
        ped_box_surface = pygame.Surface(
            (ped_box_width, ped_box_height), pygame.SRCALPHA
        )
        ped_box_surface.fill((255, 180, 180, 100))

        canvas.blit(agent_box_surface, agent_start_top_left)
        canvas.blit(agent_box_surface, agent_goal_top_left)
        canvas.blit(ped_box_surface, ped_start_top_left)
        canvas.blit(ped_box_surface, ped_goal_top_left)

        # 4. Draw Goal (Green Circle)
        goal_px = to_pixels(agent["goal"])
        pygame.draw.circle(canvas, (34, 139, 34), goal_px, int(0.2 * scale))

        # 5. Draw Obstacle Pedestrians (Red Circles with black borders)
        for ped in self._pedestrians[1:]:
            pos_px = to_pixels(ped["pos"])
            pygame.draw.circle(canvas, (220, 20, 60), pos_px, radius_px)  # Fill
            pygame.draw.circle(canvas, (0, 0, 0), pos_px, radius_px, 2)  # Border

        # 6. Draw Agent (Blue Circle with black border)
        agent_px = to_pixels(agent["pos"])
        pygame.draw.circle(canvas, (30, 144, 255), agent_px, radius_px)
        pygame.draw.circle(canvas, (0, 0, 0), agent_px, radius_px, 2)

        # 7. Output based on render_mode
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Control the framerate
            render_fps = self.metadata.get("render_fps", 10)
            self.clock.tick(render_fps)
            return None

        elif self.render_mode == "rgb_array":
            # Pygame surfaces are (X, Y, RGB). Numpy/Gym expects (Y, X, RGB).
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if hasattr(self, "window") and self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None


__all__ = ["Simple2DCross"]
