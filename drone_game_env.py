"""
Drone Missile Dodge - Custom Gymnasium Environment

This module implements a 2D drone simulation environment for reinforcement learning.
The agent controls a dual-thruster drone that must navigate through missile fields
to reach a target position.

Author: [Your Name]
Date: 2025
License: MIT
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math
import time 

# =============================================================================
# CONSTANTS - Physics and World Configuration
# =============================================================================

# Display settings
SCREEN_WIDTH = 800   # pixels
SCREEN_HEIGHT = 600  # pixels
SCALE = 20.0         # pixels per meter

# World dimensions (in meters)
WORLD_WIDTH = SCREEN_WIDTH / SCALE    # 40 meters
WORLD_HEIGHT = SCREEN_HEIGHT / SCALE  # 30 meters

# Physics constants
GRAVITY = -9.81      # m/s² (Earth gravity)
MASS = 0.7           # kg (drone mass - lighter = more agile)
TIME_STEP = 0.02     # seconds (50 FPS simulation)

# Drone specifications
DRONE_WIDTH_M = 1.5   # meters
DRONE_HEIGHT_M = 0.4  # meters
MAX_THRUST_PER_THRUSTER = 25.0  # Newtons (high = responsive)
INERTIA = (1/12) * MASS * (DRONE_WIDTH_M ** 2)  # Moment of inertia

# Missile specifications
MISSILE_WIDTH_M = 1.0   # meters (smaller = easier to dodge)
MISSILE_HEIGHT_M = 0.25 # meters
MISSILE_SPEED_M = 2.5   # m/s (horizontal speed)


class Drone2DEnv(gym.Env):
    """
    Custom Gymnasium environment for drone missile dodging.
    
    The drone must navigate from start position to target while avoiding
    missiles that travel horizontally across the screen.
    
    Observation Space (21 dimensions):
        - Drone state [0-5]: x, y, angle, vx, vy, angular_velocity
        - Target delta [6-7]: dx, dy (relative to drone)
        - Nearest missile distance [8]: scalar
        - 3 nearest missiles [9-20]: [dx, dy, vx, vy] × 3
    
    Action Space (2 dimensions):
        - Left thruster power [0]: 0.0 to 1.0
        - Right thruster power [1]: 0.0 to 1.0
    
    Rewards:
        - Progress toward target: +15.0 × distance_reduced
        - Target reached: +300.0
        - Hesitation near target: -2.0
        - Missile collision: -200.0
        - Floor/ceiling collision: -100.0
    
    Args:
        render_mode (str): "human" for visualization, None for training
        missile_count (int): Number of initial missiles (default: 10)
        spawn_close (bool): Whether to spawn missiles near drone path
    
    Example:
        >>> env = Drone2DEnv(render_mode="human", missile_count=20)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", None], "render_fps": 50}

    def __init__(self, render_mode=None, missile_count=10, spawn_close=True):
        """Initialize the drone environment."""
        super(Drone2DEnv, self).__init__()
        
        # Action space: [left_thrust, right_thrust] in range [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Better observation: drone_state(6) + target_delta(2) + nearest_dist(1) + 3 nearest missiles [dx,dy,vx,vy]*3 (12) = 21
        OBS_BOUND = 100.0
        low_obs = np.full(21, -OBS_BOUND, dtype=np.float32)
        high_obs = np.full(21, OBS_BOUND, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.target_pos = None
        self.state = None
        self.missiles = []
        self.steps_survived = 0
        self.done_reason = ""
        self.missile_count = missile_count
        self.spawn_close = spawn_close
        self.last_action = np.array([0.5, 0.5], dtype=np.float32)  # Track action changes
        self.show_blast = False  # Flag for showing blast on target hit
        self.next_wave_step = 0  # When to spawn next wave
        self.waves_spawned = 0  # Track number of waves
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
    
    # --- פונקציות עזר ---
    
    def _init_render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Drone Jetpack Dodge - AI Training")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            
            # Load images
            try:
                self.bg_image = pygame.image.load("Tehran_sky.jpg")
                self.bg_image = pygame.transform.scale(self.bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
                # Add dark overlay
                dark_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                dark_overlay.set_alpha(128)  # 50% transparency
                dark_overlay.fill((0, 0, 0))
                self.bg_image.blit(dark_overlay, (0, 0))
            except:
                self.bg_image = None
                
            try:
                self.drone_image = pygame.image.load("drone1.png")
                # Scale to 4x drone size (400% bigger)
                drone_w = int(DRONE_WIDTH_M * SCALE )
                drone_h = int(DRONE_HEIGHT_M * SCALE * 1.5)
                self.drone_image = pygame.transform.scale(self.drone_image, (drone_w, drone_h))
            except:
                self.drone_image = None
                
            try:
                self.missile_image = pygame.image.load("missile.png")
                # Scale to 4x missile size (400% bigger)
                missile_w = int(MISSILE_WIDTH_M * SCALE * 1.5)
                missile_h = int(MISSILE_HEIGHT_M * SCALE * 6)
                self.missile_image = pygame.transform.scale(self.missile_image, (missile_w, missile_h))
            except:
                self.missile_image = None
                
            try:
                self.target_image = pygame.image.load("khamn.png")
                # Scale to bigger size
                self.target_image = pygame.transform.scale(self.target_image, (40, 40))
            except:
                self.target_image = None
                
            try:
                self.blast_image = pygame.image.load("blast.png")
                # Scale blast to cover target
                self.blast_image = pygame.transform.scale(self.blast_image, (60, 60))
            except:
                self.blast_image = None

    def _create_missile(self, y_spawn):
        # Horizontal-only missiles traveling left at constant speed
        self.missiles.append({
            'x': WORLD_WIDTH * 1.05,
            'y': float(y_spawn),
            'width': MISSILE_WIDTH_M,
            'height': MISSILE_HEIGHT_M,
            'vx': -MISSILE_SPEED_M,
            'vy': 0.0,  # strictly horizontal
        })

    def _spawn_missile_wave(self):
        """Spawn a wave of 4-6 missiles spread across vertical space"""
        wave_size = random.randint(4, 6)
        drone_y = self.state[1]
        
        for i in range(wave_size):
            # Spread missiles across height, but avoid drone's exact position
            y_pos = random.uniform(1.0, WORLD_HEIGHT - 1.0)
            # Don't spawn too close to drone vertically
            while abs(y_pos - drone_y) < 2.0:
                y_pos = random.uniform(1.0, WORLD_HEIGHT - 1.0)
            
            self._create_missile(y_pos)
            # Stagger horizontally slightly
            self.missiles[-1]['x'] = WORLD_WIDTH * 1.05 + i * random.uniform(0.5, 1.5)

    def _check_collision(self, drone_state, missile):
        # Tighter collision box - need real overlap
        SAFETY_MARGIN = 0.05  # Small margin to account for discrete timesteps
        dx = abs(drone_state[0] - missile['x'])
        dy = abs(drone_state[1] - missile['y'])
        collision_x = (DRONE_WIDTH_M/2 + missile['width']/2) - SAFETY_MARGIN
        collision_y = (DRONE_HEIGHT_M/2 + missile['height']/2) - SAFETY_MARGIN
        if dx < collision_x and dy < collision_y:
            return True
        return False
        
    def _to_screen_coords(self, pos_m):
        x_m, y_m = pos_m[0], pos_m[1]
        x_p = int(x_m * SCALE + (SCREEN_WIDTH / 20)) 
        y_p = int(SCREEN_HEIGHT - (y_m * SCALE)) 
        return np.array([x_p, y_p])


    def _get_observation(self):
        MAX_VAL = 50.0
        drone_state = np.clip(self.state, -MAX_VAL, MAX_VAL).copy()
        target_delta = self.target_pos - drone_state[0:2]
        
        # Provide 3 nearest missiles for better awareness
        missile_obs = np.full(12, 100.0, dtype=np.float32)  # 3 missiles * 4 values
        nearest_dist = 100.0  # Distance to nearest missile
        
        if len(self.missiles) > 0:
            dx0 = float(drone_state[0])
            dy0 = float(drone_state[1])
            
            # Calculate distances to all missiles
            distances = []
            for idx, m in enumerate(self.missiles):
                dx = m['x'] - dx0
                dy = m['y'] - dy0
                dsq = dx * dx + dy * dy
                distances.append((dsq, idx))
            
            # Sort and take 3 nearest
            distances.sort()
            nearest_dist = math.sqrt(distances[0][0])  # Store nearest distance
            
            for i in range(min(3, len(distances))):
                _, idx = distances[i]
                m = self.missiles[idx]
                base = i * 4
                missile_obs[base + 0] = m['x'] - drone_state[0]
                missile_obs[base + 1] = m['y'] - drone_state[1]
                missile_obs[base + 2] = m['vx']
                missile_obs[base + 3] = m['vy']
        
        observation = np.concatenate((drone_state, target_delta, [nearest_dist], missile_obs))
        return observation


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_x = WORLD_WIDTH * 0.15  # Start further in
        initial_y = float(random.uniform(WORLD_HEIGHT * 0.35, WORLD_HEIGHT * 0.65))
        self.state = np.array([initial_x, initial_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_x = random.uniform(WORLD_WIDTH * 0.70, WORLD_WIDTH * 0.85)  # Far target - long journey through missiles
        target_y = random.uniform(WORLD_HEIGHT * 0.35, WORLD_HEIGHT * 0.65)
        self.target_pos = np.array([target_x, target_y], dtype=np.float32)
        self.missiles = []
        self.steps_survived = 0
        self.done_reason = ""
        self.last_action = np.array([0.5, 0.5], dtype=np.float32)  # Reset action tracker
        self.show_blast = False  # Reset blast on new episode
        self.next_wave_step = 60  # First wave after 60 steps (1.2 seconds)
        self.waves_spawned = 0
        
        if self.render_mode == "human" and self.screen is None:
             self._init_render() 
        
        # Create predictable layered pattern with guaranteed gaps
        if self.missile_count > 0:
            # Divide vertical space into layers - MORE layers for MORE missiles
            num_layers = max(7, min(self.missile_count // 2, 12))  # Scale with missile count
            layer_height = (WORLD_HEIGHT - 2.0) / (num_layers + 1)
            
            # Create gap position - this is where drone should fly
            gap_layer = random.randint(1, num_layers)
            
            positions = []
            for i in range(1, num_layers + 1):
                if i == gap_layer:
                    continue  # Skip this layer - it's the safe path
                y = 1.0 + i * layer_height
                positions.append(y)
            
            # Fill remaining missiles by adding MORE to each layer
            while len(positions) < self.missile_count:
                layer = random.choice([i for i in range(1, num_layers + 1) if i != gap_layer])
                y = 1.0 + layer * layer_height + random.uniform(-0.4, 0.4)  # More variation
                positions.append(y)
            
            # Create missiles with balanced spacing - challenging but passable
            for i, y in enumerate(positions):
                # Tighter spacing - more missiles to dodge
                x_offset = 3.0 + i * random.uniform(0.8, 1.4)
                self._create_missile(y)
                self.missiles[-1]['x'] = initial_x + x_offset
        
        self.initial_distance = float(np.linalg.norm(self.state[0:2] - self.target_pos))
        self.best_distance = self.initial_distance
        
        observation = self._get_observation()
        info = {'reason': self.done_reason, 'score': self.steps_survived}
        return observation, info


    def step(self, action):
        x, y, angle, vx, vy, v_angle = self.state
        left_thrust_cmd, right_thrust_cmd = np.clip(action, 0.0, 1.0)

        # Physics
        left_thrust = float(left_thrust_cmd) * MAX_THRUST_PER_THRUSTER
        right_thrust = float(right_thrust_cmd) * MAX_THRUST_PER_THRUSTER

        total_thrust = left_thrust + right_thrust
        force_y = total_thrust * math.cos(angle) + GRAVITY * MASS
        force_x = total_thrust * math.sin(angle)

        torque = (right_thrust - left_thrust) * (DRONE_WIDTH_M / 2)
        acc_angle = torque / INERTIA
        acc_y = force_y / MASS
        acc_x = force_x / MASS

        vx_new = vx + acc_x * TIME_STEP
        vy_new = vy + acc_y * TIME_STEP
        
        # Air resistance for better control - helps with stopping
        AIR_RESISTANCE = 0.92  # Strong damping for quick stops
        vx_new = vx_new * AIR_RESISTANCE
        vy_new = vy_new * AIR_RESISTANCE
        
        v_angle_new = (v_angle + acc_angle * TIME_STEP) * 0.93  # Even less damping for sharp turns

        V_MAX = 18.0  # High speed but air resistance makes it controllable
        ANG_V_MAX = 10.0  # Very agile rotation
        vx_new = float(np.clip(vx_new, -V_MAX, V_MAX))
        vy_new = float(np.clip(vy_new, -V_MAX, V_MAX))
        v_angle_new = float(np.clip(v_angle_new, -ANG_V_MAX, ANG_V_MAX))

        x_new = x + vx_new * TIME_STEP
        y_new = y + vy_new * TIME_STEP
        angle_new = math.atan2(math.sin(angle + v_angle_new * TIME_STEP), math.cos(angle + v_angle_new * TIME_STEP))

        self.state = np.array([x_new, y_new, angle_new, vx_new, vy_new, v_angle_new])
        self.steps_survived += 1

        # Termination & reward
        terminated = False
        reward = 0.0
        dist_to_target = float(np.linalg.norm(self.state[0:2] - self.target_pos))

        # Floor/ceiling crash - moderate penalty
        if self.state[1] < 0.5 or self.state[1] > WORLD_HEIGHT - 0.5:
            self.done_reason = "Hit Floor or Ceiling!"
            terminated = True
            reward = -100.0
            
        # Missile collision - heavy penalty
        for m in self.missiles:
            if self._check_collision(self.state, m):
                self.done_reason = "Hit by a missile!"
                terminated = True
                reward = -200.0
                break
        
        if not terminated:
            # Simple progress reward - main driving force
            if dist_to_target < self.best_distance:
                progress = self.best_distance - dist_to_target
                reward += progress * 15.0  # reward forward movement
                self.best_distance = dist_to_target
            
            # Success - big reward
            if dist_to_target < 2.0:
                self.done_reason = "Target Reached!"
                reward += 300.0
                terminated = True
                self.show_blast = True  # Show blast when target hit!
            elif dist_to_target < 3.0:
                # Strong penalty for hesitation near target
                reward -= 2.0

        # Update missiles (horizontal movement only)
        if not terminated:
            missiles_to_keep = []
            for m in self.missiles:
                m['x'] += m['vx'] * TIME_STEP
                if m['x'] > -MISSILE_WIDTH_M:
                    missiles_to_keep.append(m)
            self.missiles = missiles_to_keep
            
            # Spawn new wave of missiles periodically
            if self.steps_survived >= self.next_wave_step and self.waves_spawned < 4:
                self._spawn_missile_wave()
                self.waves_spawned += 1
                self.next_wave_step += 80  # Next wave every 80 steps (~1.6 seconds)

        truncated = False
        
        if self.render_mode == "human":
            self.render()
        
        observation = self._get_observation()
        info = {'reason': self.done_reason, 'score': self.steps_survived}
        
        return observation, reward, terminated, truncated, info


    def render(self):
        """ מצייר את המצב הנוכחי על מסך Pygame עם תמונות. """
        if self.render_mode != "human":
            return

        if self.screen is None:
            self._init_render() 

        pygame.event.pump() 
        
        # Draw background
        if self.bg_image:
            self.screen.blit(self.bg_image, (0, 0))
        else:
            self.screen.fill((20, 20, 40))
        
        x, y, angle, _, _, _ = self.state
        center_m = np.array([x, y])
        center_p = self._to_screen_coords(center_m)
        
        # Draw target (green point)
        target_center_p = self._to_screen_coords(self.target_pos)
        if self.target_image:
            target_rect = self.target_image.get_rect(center=target_center_p)
            self.screen.blit(self.target_image, target_rect)
        else:
            pygame.draw.circle(self.screen, (0, 200, 0), target_center_p, 10)
        
        # Draw blast on top of target if hit
        if self.show_blast and self.blast_image:
            blast_rect = self.blast_image.get_rect(center=target_center_p)
            self.screen.blit(self.blast_image, blast_rect)
        
        # Draw drone
        if self.drone_image:
            angle_deg = -math.degrees(angle)
            rotated_drone = pygame.transform.rotate(self.drone_image, angle_deg)
            new_rect = rotated_drone.get_rect(center=center_p)
            self.screen.blit(rotated_drone, new_rect)
        else:
            # Fallback to colored rectangle
            DRONE_W_P = DRONE_WIDTH_M * SCALE
            DRONE_H_P = DRONE_HEIGHT_M * SCALE
            drone_body = pygame.Surface((DRONE_W_P, DRONE_H_P), pygame.SRCALPHA)
            drone_body.fill((0, 0, 255))
            angle_deg = -math.degrees(angle)
            rotated_drone = pygame.transform.rotate(drone_body, angle_deg)
            new_rect = rotated_drone.get_rect(center=center_p)
            self.screen.blit(rotated_drone, new_rect)
        
        # Draw missiles
        for missile in self.missiles:
            m_center_p = self._to_screen_coords(np.array([missile['x'], missile['y']]))
            if self.missile_image:
                m_rect = self.missile_image.get_rect(center=m_center_p)
                self.screen.blit(self.missile_image, m_rect)
            else:
                # Fallback to colored rectangle
                m_rect = pygame.Rect(0, 0, missile['width'] * SCALE, missile['height'] * SCALE)
                m_rect.center = m_center_p
                pygame.draw.rect(self.screen, (255, 0, 0), m_rect)

        # Draw score and status
        score_text = self.font.render(f"Score: {self.steps_survived}", True, (255, 255, 255))
        reason_text = self.font.render(f"Status: {self.done_reason}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(reason_text, (10, 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None