# Drone Stable Hover RL

Deep Reinforcement Learning for Wind-Resistant Drone Hovering Control

## Project Background

Wind interference is a major factor affecting hovering stability in outdoor drone operations. Traditional PID controllers often require retuning when wind conditions change, while deep reinforcement learning can automatically learn optimal control strategies through environment interaction. This project implements PPO, TD3, and DQN algorithms in the Isaac Lab simulation environment, comparing their performance in wind-resistant hovering tasks.

## Technical Implementation Details

### Environment Modeling

**State Space (18 dimensions)**
- Position: [x, y, z] (3D)
- Attitude angles: [roll, pitch, yaw] (3D)
- Linear velocity: [vx, vy, vz] (3D)
- Angular velocity: [wx, wy, wz] (3D)
- Position error: [x_error, y_error, z_error] (3D)
- Target position: [target_x, target_y, target_z] (3D)

**Action Space (4 dimensions)**
- Four propeller RPM control: [rpm1, rpm2, rpm3, rpm4]
- RPM range: 4500-5500
- Action mapping: [-1,1] → [4500,5500]

**Wind Disturbance Model**
```python
# Random wind strength generation (0.5N - 2.0N)
wind_strength = np.random.uniform(min_wind, max_wind)

# Random wind direction changes (-π to π)
wind_direction = np.random.uniform(-np.pi, np.pi)

# Wind strength time variation (update every 300 steps)
if steps % 300 == 0:
    wind_strength *= (1 + random_variation)
```

### Reward Function Design

**Position Reward**
```python
position_error = np.linalg.norm(current_pos - target_pos)
position_reward = -position_error * 10.0
```

**Velocity Stability Reward**
```python
velocity_penalty = -np.sum(np.abs(linear_velocity)) * 2.0
angular_velocity_penalty = -np.sum(np.abs(angular_velocity)) * 1.0
```

**Attitude Stability Reward**
```python
attitude_penalty = -(abs(roll) + abs(pitch)) * 5.0
```

**Total Reward**
```python
total_reward = position_reward + velocity_penalty + angular_velocity_penalty + attitude_penalty
```

### Algorithm Implementation Comparison

#### PPO (Successful Solution)
- **Network Architecture**: Actor-Critic dual networks
- **Actor Network**: Input 18D state, output 4D action mean and standard deviation
- **Critic Network**: Input 18D state, output state value
- **Advantage Estimation**: GAE (Generalized Advantage Estimation)
- **Policy Update**: Importance sampling ratio clipping
- **Hyperparameters**: Learning rate 0.0003, batch size 64, update frequency 128

#### TD3
- **Network Architecture**: Dual Q-networks + Policy network
- **Delayed Update**: Policy network updates every 2 Q-network updates
- **Target Policy Smoothing**: Add noise to target actions
- **Hyperparameters**: Learning rate 0.0003, batch size 100, update frequency 2

#### DQN
- **Network Architecture**: Deep Q-network
- **Experience Replay**: Buffer size 10000
- **Target Network**: Updates every 100 steps
- **Exploration Strategy**: ε-greedy, decay from 1.0 to 0.01
- **Hyperparameters**: Learning rate 0.0001, batch size 32

### Training Process

**Training Parameters**
- Maximum episodes: 1000
- Maximum steps per episode: 5000
- Time step: 0.02 seconds
- Target position: (0, 0, 1) meters

**Convergence Results**
- PPO: Converges around 500-800 episodes, final average reward -50 to -100
- TD3: Converges around 800-1000 episodes, final average reward -80 to -150
- DQN: Slower convergence, final average reward -120 to -200

**Wind Resistance Performance Test**
```python
# Test performance under different wind strengths
wind_speeds = [0.5, 1.0, 1.5, 2.0]  # N
for wind_speed in wind_speeds:
    # Record position error and recovery time
    position_error = measure_position_error()
    recovery_time = measure_recovery_time()
```

### Key Code Structure

**Environment Reset**
```python
def reset(self):
    # Random initial position offset
    initial_height = self.target_position[2] + np.random.uniform(-0.1, 0.1)
    initial_x = np.random.uniform(-0.1, 0.1)
    initial_y = np.random.uniform(-0.1, 0.1)
    
    # Reset wind
    self._reset_wind()
    
    return self._get_state()
```

**State Acquisition**
```python
def _get_state(self):
    # Get physical states
    pos = self.drone.get_world_pose()[0]
    lin_vel = self.drone.get_linear_velocity()
    ang_vel = self.drone.get_angular_velocity()
    
    # Calculate position error
    position_error = pos - self.target_position
    
    # Combine state vector
    state = np.concatenate([pos, orientation, lin_vel, ang_vel, position_error])
    return state
```

**Action Execution**
```python
def step(self, action):
    # Map [-1,1] actions to RPM
    rpm_action = min_rpm + (action + 1) * 0.5 * rpm_range
    rpm_action = np.clip(rpm_action, min_rpm, max_rpm)
    
    # Apply wind disturbance
    wind_force = self._calculate_wind_force()
    
    # Execute action
    self.drone.apply_force(wind_force)
    self.drone.set_propeller_rpms(rpm_action)
    
    # Get new state and reward
    next_state = self._get_state()
    reward = self._calculate_reward()
    done = self._check_termination()
    
    return next_state, reward, done, {}
```

## Experimental Results

### Training Performance Comparison
| Algorithm | Convergence Episodes | Final Avg Reward | Wind Resistance Success Rate | Avg Recovery Time |
|-----------|---------------------|------------------|------------------------------|-------------------|
| PPO       | 500-800            | -50 ~ -100       | 85%                          | 2.5s              |
| TD3       | 800-1000           | -80 ~ -150       | 70%                          | 3.8s              |
| DQN       | >1000              | -120 ~ -200      | 45%                          | 5.2s              |

### Wind Adaptation Analysis
- **Position Error**: Under 2.0N wind, PPO algorithm can control position error within 0.3m
- **Recovery Capability**: Average recovery time from wind disturbance to stable state is 2.5s for PPO
- **Stability**: Continuous hovering success rate reaches 85% for PPO, only 45% for DQN

## Project File Structure

```
Drone-Stable-Hover-RL/
├── PPO/
│   ├── drone_PPO.py          # PPO main program, includes training and evaluation
│   ├── models/               # Trained model files
│   └── training_logs/        # Training process log data
├── TD3/
│   ├── drone_TD3.py          # TD3 algorithm implementation
│   └── models/               # TD3 model files
├── DQN/
│   ├── drone_DQN.py          # DQN algorithm implementation
│   └── models/               # DQN model files
├── outputs/
│   ├── plots/                # Training curves and performance analysis plots
│   └── logs/                 # Runtime logs and error records
└── visualization.py           # Data visualization tools
```

## Running Instructions

### Environment Configuration
```bash
# Install dependencies
pip install torch numpy matplotlib scipy

# Ensure Isaac Lab environment is configured
# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
```

### Training Commands
```bash
# PPO training
cd PPO
python drone_PPO.py --max_episodes 1000 --enable_wind True

# TD3 training  
cd TD3
python drone_TD3.py --max_episodes 1000 --enable_wind True

# DQN training
cd DQN
python drone_DQN.py --max_episodes 1000 --enable_wind True
```

### Model Evaluation
```bash
# Evaluate trained model
python evaluate_model.py --model_path models/best_model.pt --num_episodes 10 --enable_wind True
```

## Technical Challenges and Solutions

### 1. Wind Modeling
**Problem**: How to design realistic wind disturbance models
**Solution**: Implement random wind strength and direction generation with time-varying characteristics

### 2. Reward Function Design
**Problem**: How to balance position accuracy and stability
**Solution**: Multi-dimensional reward function considering position, velocity, and attitude stability

### 3. Training Stability
**Problem**: Unstable training in continuous action spaces
**Solution**: PPO importance sampling ratio clipping, TD3 target policy smoothing

### 4. Wind Resistance Performance Evaluation
**Problem**: How to quantify wind resistance capability
**Solution**: Design multi-dimensional evaluation metrics including recovery time and position error

## Project Value

1. **Algorithm Comparison**: First comparison of PPO, TD3, and DQN algorithms in drone wind resistance tasks
2. **Environment Design**: Designed drone simulation environment with random wind disturbances
3. **Performance Evaluation**: Established comprehensive wind resistance performance evaluation system
4. **Engineering Practice**: Provided reproducible deep reinforcement learning training framework

## Contact Information

- GitHub: [NUSNiuMu](https://github.com/NUSNiuMu)
- Project Repository: [Drone-Stable-Hover-RL](https://github.com/NUSNiuMu/Drone-Stable-Hover-RL)
