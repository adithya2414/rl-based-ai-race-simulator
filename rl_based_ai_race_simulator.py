import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import matplotlib.pyplot as plt

# ===================== DEVICE =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ===================== GAME CONFIG =====================
WIDTH, HEIGHT = 1000, 600
FPS = 60

TRACK_CENTER_Y = HEIGHT // 2
TRACK_WIDTH = 220

MAX_SPEED = 8.0
ACCEL = 0.15
STEER_GAIN = 0.04
FRICTION = 0.02

# ===================== PPO CONFIG =====================
STATE_DIM = 6
ACTION_DIM = 2

GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
LR = 1.5e-4
EPOCHS = 5
TOTAL_EPISODES = 500

# ===================== TRACK =====================
def track_center(x):
    return TRACK_CENTER_Y + 80 * math.sin(x * 0.009) + 20 * math.cos(x * 0.012)



# ===================== CAR =====================
class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 50.0
        self.y = float(track_center(self.x))
        self.angle = 0.0
        self.speed = 0.0
        self.path = []
        self.crashed = False
        self.crash_timer = 0

    def update(self, steer, throttle):
        if self.crashed:
            return

        self.speed += throttle * ACCEL
        self.speed = np.clip(self.speed, 0, MAX_SPEED)
        self.speed *= (1 - FRICTION)

        self.angle += steer * STEER_GAIN
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        self.path.append((int(self.x), int(self.y)))

    def off_track(self):
        return abs(self.y - track_center(self.x)) > TRACK_WIDTH // 2

# ===================== PPO NETWORK =====================
class PPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, ACTION_DIM)
        self.value = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def forward(self, x):
        x = self.shared(x)
        return self.mu(x), self.log_std.exp(), self.value(x)

# ===================== AGENT =====================
class Agent:
    def __init__(self):
        self.model = PPO().to(DEVICE).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.states, self.actions = [], []
        self.log_probs, self.rewards = [], []
        self.values, self.dones = [], []

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mu, std, value = self.model(state_t)

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)

        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu(), value.squeeze(0).cpu()

    def store(self, s, a, lp, r, v, d):
        self.states.append(torch.tensor(s, dtype=torch.float32))
        self.actions.append(torch.tensor(a, dtype=torch.float32))
        self.log_probs.append(lp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(d)

    def update(self):
        if len(self.states) == 0:
            return

        states = torch.stack(self.states).to(DEVICE)
        actions = torch.stack(self.actions).to(DEVICE)
        old_log_probs = torch.stack(self.log_probs).to(DEVICE)
        values = torch.stack(self.values).to(DEVICE).squeeze()

        returns, advs = [], []
        gae, next_value = 0, 0

        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + GAMMA * next_value * (1 - self.dones[i]) - values[i]
            gae = delta + GAMMA * LAMBDA * (1 - self.dones[i]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]

        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        advs = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(EPOCHS):
            mu, std, value = self.model(states)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advs

            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * (returns - value.squeeze()).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

# ===================== METRICS STORAGE =====================
episode_rewards = []
lap_times = []
best_lap_history = []
episode_lengths = []
mean_speeds = []
max_speeds = []
steer_means = []
throttle_means = []
crash_flags = []

# ===================== MAIN =====================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PPO Racing – Ghost & Metrics")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

agent = Agent()
car = Car()

best_lap_time = float("inf")
ghost_path = None

for episode in range(1, TOTAL_EPISODES + 1):
    car.reset()
    steps = 0
    total_reward = 0
    speeds, steers, throttles = [], [], []
    crashed = 0

    for step in range(1500):
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        state = np.array([
            car.y - track_center(car.x),
            car.angle,
            car.speed,
            car.x / WIDTH,
            math.sin(car.x * 0.004),
            math.cos(car.x * 0.004)
        ], dtype=np.float32)

        action, logp, val = agent.select_action(state)
        steer = np.clip(action[0], -1, 1)
        throttle = np.clip(action[1], 0, 1)

        car.update(steer, throttle)

        reward = car.speed * 0.3
        done = False

        speeds.append(car.speed)
        steers.append(abs(steer))
        throttles.append(throttle)

        if car.off_track():
            reward -= 80
            crashed = 1
            done = True

        if car.x > WIDTH - 60:
            lap_time = steps / FPS
            reward += 300 + (50 / lap_time)
            lap_times.append(lap_time)
            done = True

            if lap_time < best_lap_time:
                best_lap_time = lap_time
                ghost_path = copy.deepcopy(car.path)

        agent.store(state, action, logp, reward, val, done)
        total_reward += reward
        steps += 1

        screen.fill((25, 25, 25))
        pts = [(x, int(track_center(x))) for x in range(0, WIDTH, 5)]
        pygame.draw.lines(screen, (70, 70, 70), False, pts, TRACK_WIDTH)

        if ghost_path:
            pygame.draw.lines(screen, (0,150,255), False, ghost_path, 2)

        if len(car.path) > 2:
            pygame.draw.lines(screen, (0,255,0), False, car.path, 2)

        pygame.draw.rect(screen, (255,0,0), (int(car.x)-10, int(car.y)-5, 20, 10))
        screen.blit(font.render(f"Ep {episode} Best Lap {best_lap_time:.2f}s", True, (255,255,255)), (10,10))
        pygame.display.flip()

        if done:
            break

    agent.update()

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    mean_speeds.append(np.mean(speeds))
    max_speeds.append(np.max(speeds))
    steer_means.append(np.mean(steers))
    throttle_means.append(np.mean(throttles))
    crash_flags.append(crashed)
    best_lap_history.append(best_lap_time)

    print(f"Episode {episode:03d} | Reward {total_reward:.1f} | Best Lap {best_lap_time:.2f}s")

pygame.quit()

# ===================== PLOTS =====================
def moving_avg(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(18,20))

plt.subplot(5,4,1); plt.plot(episode_rewards); plt.plot(moving_avg(episode_rewards)); plt.title("Episode Reward")
plt.subplot(5,4,2); plt.plot(lap_times); plt.title("Lap Times")
plt.subplot(5,4,3); plt.plot(best_lap_history); plt.title("Best Lap Improvement")
plt.subplot(5,4,4); plt.plot(episode_lengths); plt.title("Episode Length")
plt.subplot(5,4,5); plt.plot(mean_speeds); plt.title("Mean Speed")
plt.subplot(5,4,6); plt.plot(max_speeds); plt.title("Max Speed")
plt.subplot(5,4,7); plt.plot(steer_means); plt.title("Steering Smoothness")
plt.subplot(5,4,8); plt.plot(throttle_means); plt.title("Throttle Usage")
plt.subplot(5,4,9); plt.plot(moving_avg(crash_flags)); plt.title("Crash Rate")
plt.subplot(5,4,10); plt.plot(np.array(episode_rewards)/np.array(episode_lengths)); plt.title("Reward/Step")

plt.tight_layout()
plt.show()
