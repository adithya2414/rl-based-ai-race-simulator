# rl-based-ai-race-simulator
rl based ai race simulator
# PPO Racing AI
# PPO Racing AI with Ghost Car & Multi-Agent Extensions

This project implements a continuous-control racing environment using
Proximal Policy Optimization (PPO) with PyTorch and Pygame.

## Features
- Continuous steering & throttle
- PPO (Actor-Critic)
- Ghost car (best lap replay)
- Crash detection & animation
- Speedometer HUD
- Episode metrics & visualization
- Ready for SAC / TD3 comparison
- Human vs AI keyboard extension ready
- Multi-AI racing grid (scalable)

## Metrics Tracked
- Episode reward
- Lap time optimization
- Crash rate
- Speed statistics
- Control smoothness

## Future Work
- SAC / TD3 implementation
- Curriculum learning
- Domain randomization
- Realistic vehicle dynamics
- Multi-agent competition

A real-time racing simulation using PPO reinforcement learning with
continuous steering and throttle.

## Features
- PPO (Actor-Critic)
- Ghost car replay
- Lap time optimization
- Crash detection
- Live HUD
- Training metrics visualization

## How to Run

```bash
git clone https://github.com/<your-username>/ppo-racing-ai.git
cd ppo-racing-ai
pip install -r requirements.txt
python main.py
