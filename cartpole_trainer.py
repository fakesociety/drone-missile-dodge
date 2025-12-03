import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math
import time # ודא ש-time מיובא

# --- 1. הגדרות קבועים של המשחק ---

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCALE = 20.0 

WORLD_WIDTH = SCREEN_WIDTH / SCALE
WORLD_HEIGHT = SCREEN_HEIGHT / SCALE

# קבועים פיזיקליים
GRAVITY = -9.81
MASS = 1.0
TIME_STEP = 0.02 # 50Hz (1 / 50)

# הגדרות רחפן
DRONE_WIDTH_M = 2.0
DRONE_HEIGHT_M = 0.5
MAX_THRUST = 15.0  
THRUST_HOVER = (-GRAVITY * MASS) / 2.0 
INERTIA = (1/12) * MASS * (DRONE_WIDTH_M ** 2) 

# הגדרות הטילים
MISSILE_WIDTH_M = 1.5
MISSILE_HEIGHT_M = 0.4
MISSILE_SPEED_M = 4.0
MISSILE_SPAWN_PROB = 0.02 

class Drone2DEnv(gym.Env):
    """
    סביבת Gymnasium מותאמת אישית למשחק "Jetpack Dodge" המלא.
    """
    metadata = {"render_modes": ["human", None], "render_fps": 50}

    def __init__(self, render_mode=None):
        super(Drone2DEnv, self).__init__()
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        low_obs = np.full(12, -np.inf, dtype=np.float32)
        high_obs = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.target_pos = None
        self.state = None 
        self.missiles = [] 
        self.steps_survived = 0
        self.done_reason = ""
        self.prev_dist = 0 
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
    
    # --- פונקציות עזר ---
    
    def _init_render(self):
        """ מפעיל את Pygame בפעם הראשונה """
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Drone Jetpack Dodge - AI Training")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

    def _create_missile(self):
        """ יוצר טיל חדש ב-Y אקראי בצד ימין """
        y_spawn = random.uniform(MISSILE_HEIGHT_M, WORLD_HEIGHT - MISSILE_HEIGHT_M)
        self.missiles.append({
            'x': WORLD_WIDTH * 0.9, 'y': y_spawn,
            'w': MISSILE_WIDTH_M, 'h': MISSILE_HEIGHT_M,
            'speed': MISSILE_SPEED_M
        })

    def _check_collision(self, drone_state, missile):
        """ בדיקת התנגשות """
        dx = abs(drone_state[0] - missile['x'])
        dy = abs(drone_state[1] - missile['y'])
        if dx < (DRONE_WIDTH_M/2 + missile['w']/2) and dy < (DRONE_HEIGHT_M/2 + missile['h']/2):
            return True
        return False
        
    def _to_screen_coords(self, pos_m):
        """ ממיר מיקום מטרים לקואורדינטות פיקסלים """
        x_m, y_m = pos_m[0], pos_m[1]
        x_p = int(x_m * SCALE + (SCREEN_WIDTH / 20)) 
        y_p = int(SCREEN_HEIGHT - (y_m * SCALE)) 
        return np.array([x_p, y_p])


    def _get_observation(self):
        """ מחבר את הנתונים הפיזיקליים (12 ערכים) ל-AI """
        
        MAX_VAL = 50.0 
        drone_state = np.clip(self.state, -MAX_VAL, MAX_VAL).copy() 
        target_delta = self.target_pos - drone_state[0:2]
        missile_obs = np.full(4, 100.0, dtype=np.float32)
        sorted_missiles = sorted(self.missiles, key=lambda m: abs(m['x'] - drone_state[0]))
        if len(sorted_missiles) > 0:
            missile1 = sorted_missiles[0]
            missile_obs[0] = missile1['x'] - drone_state[0] 
            missile_obs[1] = missile1['y'] - drone_state[1]
        if len(sorted_missiles) > 1:
            missile2 = sorted_missiles[1]
            missile_obs[2] = missile2['x'] - drone_state[0]
            missile_obs[3] = missile2['y'] - drone_state[1]
        observation = np.concatenate((drone_state, target_delta, missile_obs))
        return observation


    def reset(self, seed=None, options=None):
        """ מאפס את המשחק למשימת "התחמקות וניווט" המלאה """
        super().reset(seed=seed)
        initial_x = WORLD_WIDTH * 0.1 
        initial_y = WORLD_HEIGHT / 2.0
        self.state = np.array([initial_x, initial_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_x = random.uniform(WORLD_WIDTH * 0.7, WORLD_WIDTH * 0.9)
        target_y = random.uniform(WORLD_HEIGHT * 0.2, WORLD_HEIGHT * 0.8)
        self.target_pos = np.array([target_x, target_y], dtype=np.float32)
        self.missiles = [] 
        self.steps_survived = 0
        self.done_reason = ""
        self.prev_dist = np.linalg.norm(self.state[0:2] - self.target_pos)
        
        if self.render_mode == "human" and self.screen is None:
             self._init_render() 
        
        observation = self._get_observation()
        info = {'reason': self.done_reason, 'score': self.steps_survived}
        
        return observation, info


    def step(self, action):
        """ המנוע הפיזיקלי המלא """
        
        x, y, angle, vx, vy, v_angle = self.state
        left_thrust_cmd, right_thrust_cmd = action

        base_thrust = THRUST_HOVER 
        max_action_thrust = MAX_THRUST - base_thrust
        left_thrust = base_thrust + (left_thrust_cmd * max_action_thrust)
        right_thrust = base_thrust + (right_thrust_cmd * max_action_thrust)
        
        total_thrust = left_thrust + right_thrust
        force_y = total_thrust * math.cos(angle) + GRAVITY * MASS
        force_x = total_thrust * math.sin(angle)
        
        torque = (right_thrust - left_thrust) * (DRONE_WIDTH_M / 2) 
        acc_angle = torque / INERTIA
        acc_y = force_y / MASS
        acc_x = force_x / MASS
        
        vx_new = vx + acc_x * TIME_STEP
        vy_new = vy + acc_y * TIME_STEP
        v_angle_new = v_angle + acc_angle * TIME_STEP
        x_new = x + vx_new * TIME_STEP
        y_new = y + vy_new * TIME_STEP
        angle_new = angle + v_angle_new * TIME_STEP
        angle_new = math.atan2(math.sin(angle_new), math.cos(angle_new))
        
        self.state = np.array([x_new, y_new, angle_new, vx_new, vy_new, v_angle_new])
        self.steps_survived += 1

        # --- לוגיקת סיום ותגמול (מעודד התקדמות) ---
        
        terminated = False
        reward = 0.0
        dist_to_target = np.linalg.norm(self.state[0:2] - self.target_pos)

        if self.state[1] < 0.5 or self.state[1] > WORLD_HEIGHT - 0.5:
            self.done_reason = "Hit Floor or Ceiling!"
            terminated = True
            reward = -100.0 
        elif abs(angle_new) > math.radians(60): 
            self.done_reason = "Tilted too much!"
            terminated = True
            reward = -100.0
            
        for m in self.missiles:
            if self._check_collision(self.state, m):
                self.done_reason = "Hit by a missile!"
                terminated = True
                reward = -100.0
                break
        
        if not terminated:
            reward_shaping = (self.prev_dist - dist_to_target) * 10.0
            reward += reward_shaping
            self.prev_dist = dist_to_target
            
            reward -= (self.state[2] ** 2) * 0.5
            reward -= abs(self.state[5]) * 0.2

            if dist_to_target < 1.0: 
                self.done_reason = "Target Reached!"
                reward += 500.0 
                terminated = True
        
        if not terminated:
            missiles_to_keep = []
            for m in self.missiles:
                m['x'] -= MISSILE_SPEED_M * TIME_STEP
                if m['x'] > -MISSILE_WIDTH_M: 
                    missiles_to_keep.append(m)
            self.missiles = missiles_to_keep
            
            if self.steps_survived > 50 and random.random() < MISSILE_SPAWN_PROB:
                 self._create_missile()

        truncated = False 
        
        if self.render_mode == "human":
            self.render() 
        
        observation = self._get_observation()
        info = {'reason': self.done_reason, 'score': self.steps_survived}

        # --- [תיקון סנכרון זמן קריטי] ---
        # אם אנחנו *לא* במצב גרפי (כלומר, באימון מהיר)
        # נוסיף השהייה קטנה כדי לכפות את קצב הזמן האמיתי
        if self.render_mode is None:
            time.sleep(TIME_STEP) # TIME_STEP = 0.02 שניות
        
        return observation, reward, terminated, truncated, info


    def render(self):
        """ מצייר את המצב הנוכחי על מסך Pygame. """
        if self.render_mode != "human":
            return

        if self.screen is None:
            self._init_render() 

        pygame.event.pump() 
        
        self.screen.fill((20, 20, 40)) 
        
        x, y, angle, _, _, _ = self.state
        center_m = np.array([x, y])
        center_p = self._to_screen_coords(center_m)
        
        target_center_p = self._to_screen_coords(self.target_pos)
        pygame.draw.circle(self.screen, (0, 200, 0), target_center_p, 10) 
        
        DRONE_W_P = DRONE_WIDTH_M * SCALE
        DRONE_H_P = DRONE_HEIGHT_M * SCALE
        drone_body = pygame.Surface((DRONE_W_P, DRONE_H_P), pygame.SRCALPHA)
        drone_body.fill((0, 0, 255))
        angle_deg = -math.degrees(angle) 
        rotated_drone = pygame.transform.rotate(drone_body, angle_deg)
        new_rect = rotated_drone.get_rect(center=center_p)
        self.screen.blit(rotated_drone, new_rect)
        
        for missile in self.missiles:
            m_center_p = self._to_screen_coords(np.array([missile['x'], missile['y']]))
            m_rect = pygame.Rect(0, 0, missile['w'] * SCALE, missile['h'] * SCALE)
            m_rect.center = m_center_p
            pygame.draw.rect(self.screen, (255, 0, 0), m_rect)

        score_text = self.font.render(f"Score: {self.steps_survived}", True, (255, 255, 255))
        reason_text = self.font.render(f"Status: {self.done_reason}", True, (255, 200, 200))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(reason_text, (10, 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """ מכבה את Pygame בצורה מסודרת """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None