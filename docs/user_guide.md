# User Guide: Rotating Space Habitat Simulation

## Introduction and Overview

The Rotating Space Habitat Simulation is a Python-based framework for modeling the dynamics of a rotating space habitat in microgravity. It simulates rigid body physics, disturbances from crew and cargo movements, and active vibration control using a system of rim-mounted water tanks. The simulation includes reinforcement learning (RL) capabilities to train an agent that autonomously stabilizes the habitat by redistributing water mass.

### Key Features

- **Physics Simulation**: Rigid body dynamics with quaternion-based orientation tracking, inertia tensor computation, and RK4 integration.
- **Disturbances**: Prescribed or stochastic crew/cargo movements, micro-impacts, and mass imbalances.
- **Control System**: 36 rim water tanks connected via manifolds for mass redistribution, controlled by a Soft Actor-Critic (SAC) RL agent.
- **Sensors**: 6 three-axis accelerometers and noisy mass trackers for state observation.
- **RL Training**: Integration with Stable Baselines3 for training and evaluation.
- **Database Logging**: SQLite-based storage of experiments, episodes, and timesteps for analysis.
- **CLI Tools**: Command-line interfaces for training, simulation, and analysis.
- **Extensibility**: Support for different habitat geometries (cylinder, ring, toroid) and future vibration modes.

### Use Cases

- Research in spacecraft dynamics and vibration control.
- RL algorithm development for continuous control tasks.
- Educational simulations of artificial gravity and orbital mechanics.

### Architecture Overview

The simulation is built around a modular architecture:

- **Core**: Quaternion math, inertia computation, RK4 integrator.
- **Geometry**: Habitat shapes (cylinder, ring, toroid) with structural inertia.
- **Dynamics**: Rigid body equations with variable inertia.
- **Actuators**: Tank system with hybrid manifolds.
- **Sensors**: Accelerometer and mass tracker models.
- **Disturbances**: Mass schedules and stochastic events.
- **Environment**: Gymnasium-compatible RL environment.
- **Control**: SAC agent and training pipeline.
- **Database**: Experiment logging and querying.
- **Scripts**: CLI entry points.

Data flows from configuration to simulation engine, through physics integration, to RL agent actions, with optional logging to database.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Windows, macOS, or Linux

### Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### Install the Package

Install core dependencies (numpy, scipy):

```bash
pip install -e .
```

For RL training:

```bash
pip install -e ".[rl]"
```

For database and visualization:

```bash
pip install -e ".[db,viz]"
```

For everything:

```bash
pip install -e ".[all]"
```

### Windows Long Paths

If installing torch fails due to path length, enable long paths:

```powershell
# Run as Administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

Restart your terminal and retry.

### Verify Installation

Run tests:

```bash
pytest
```

All 113 tests should pass.

## Quick Start

### Run Demos

Execute the quick simulation script:

```bash
python scripts/quick_sim.py
```

This runs four demos:
- Torque-free conservation check
- Mass imbalance whirl
- Tank correction
- Random Gymnasium agent episode

### Train an Agent

Train a SAC agent for 500,000 timesteps:

```bash
habitat-train --timesteps 500000 --n-envs 4 --log-dir ./runs/my_run
```

Record to database:

```bash
habitat-train --timesteps 500000 --db habitat.db --experiment-name my_run
```

### Analyze Results

Plot training metrics:

```bash
python -m habitat_sim.scripts.analyse_experiment --db habitat.db --experiment-id 1
```

## Configuration

All parameters are defined in `habitat_sim/config.py` as dataclasses. The `ExperimentConfig` bundles sub-configs and supports JSON serialization.

### Reference Configuration

Load defaults:

```python
from habitat_sim.config import reference_config

cfg = reference_config()
```

### Sub-Configurations

- **HabitatConfig**: Geometry (shape, radius, length, wall density).
- **SectorConfig**: Angular and axial sector grid (n_angular=12, n_axial=3).
- **TankConfig**: Tank count, capacity, flow rates.
- **MotorConfig**: Spin motor torque profiles.
- **SensorConfig**: Accelerometer count and noise.
- **SimulationConfig**: Timesteps, durations.
- **RLConfig**: SAC hyperparameters, curriculum.
- **StochasticConfig**: Poisson crew and micro-impacts.

### Modifying Config

Override fields directly:

```python
cfg.rl.total_timesteps = 250_000
cfg.stochastic.poisson_crew = True
```

Serialize to JSON:

```python
print(cfg.to_json())
```

Load from JSON:

```python
cfg = ExperimentConfig.from_dict(json.loads(json_str))
```

### Toroid Geometry Example

Switch to toroid shape:

```python
from habitat_sim.config import reference_config
from dataclasses import replace

cfg = reference_config()
cfg.habitat = replace(cfg.habitat, shape="toroid", minor_radius=5.0)
cfg.sectors = SectorConfig(n_angular=12, n_axial=1)
cfg.tanks = replace(cfg.tanks, n_tanks_per_station=12, n_stations=1)
cfg.sensors = replace(cfg.sensors, n_accelerometers=2)
```

## Running Simulations and Demos

### CLI: habitat-run

Run demos or evaluate models:

```bash
# Run all demos
habitat-run --demo all

# Run specific demo
habitat-run --demo imbalance

# Evaluate trained model
habitat-run --model ./runs/best_model/best_model.zip --episodes 10
```

Available demos:
- `torque-free`: Conservation laws check.
- `imbalance`: Mass offset whirl.
- `tank`: Active correction.
- `random-agent`: Gymnasium episode.

### Programmatic Simulation

```python
from habitat_sim.config import reference_config
from habitat_sim.environment.habitat_env import HabitatEnv
import numpy as np

cfg = reference_config()
env = HabitatEnv(config=cfg)
obs, _ = env.reset(seed=42)

for _ in range(100):
    action = np.zeros(36)  # No control
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final nutation: {info['nutation_angle_deg']:.2f}°")
env.close()
```

### Simulation Engine

Direct engine access:

```python
from habitat_sim.simulation.engine import SimulationEngine

engine = SimulationEngine(cfg)
obs = engine.reset(seed=42)

for _ in range(100):
    obs, info = engine.step(np.zeros(36))

print(f"CM offset: {engine.get_cm_offset_magnitude():.4f} m")
```

## Training Agents

### CLI Training

```bash
habitat-train --timesteps 500000 --n-envs 4 --log-dir ./runs/my_run
```

Options:
- `--config`: JSON config file.
- `--timesteps`: Total training steps.
- `--n-envs`: Parallel environments.
- `--log-dir`: Output directory.
- `--seed`: Random seed.
- `--no-curriculum`: Disable progressive difficulty.
- `--db`: SQLite database path.
- `--experiment-name`: Experiment name for DB.

### Programmatic Training

```python
from habitat_sim.config import reference_config
from habitat_sim.control.training import run_training

cfg = reference_config()
cfg.rl.total_timesteps = 500_000
cfg.rl.n_envs = 4

model = run_training(cfg)
```

With database logging:

```python
from habitat_sim.database.recorder import ExperimentRecorder

with ExperimentRecorder("habitat.db", "my_experiment", cfg) as rec:
    model = run_training(cfg, recorder=rec)
```

### Curriculum Learning

Enabled by default, ramps imbalance mass: 0kg → 50kg → 150kg → 200kg over training.

### Evaluation

```python
from habitat_sim.control.training import evaluate_agent

results = evaluate_agent("./runs/best_model/best_model.zip", cfg, n_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
print(f"Mean nutation: {results['mean_nutation_deg']:.3f}°")
```

## Analyzing Experiments

### Database Queries

```python
from habitat_sim.database.queries import list_experiments, get_reward_curve

experiments = list_experiments("habitat.db")
curve = get_reward_curve("habitat.db", experiment_id=1)
```

### CLI Analysis

```bash
python -m habitat_sim.scripts.analyse_experiment --db habitat.db --experiment-id 1 --out-dir ./plots
```

Generates plots:
- Episode reward over time.
- Final nutation per episode.
- H-violation count.

### Custom Queries

```python
from habitat_sim.database.queries import get_nutation_curve, get_conservation_summary

nutation = get_nutation_curve("habitat.db", 1)
conservation = get_conservation_summary("habitat.db", 1)
```

## API Reference

### Key Classes

#### `ExperimentConfig`

Top-level configuration.

- `from_dict(d)`: Load from dict.
- `to_json()`: Serialize to JSON.

#### `SimulationEngine`

Core simulator.

- `__init__(config)`: Initialize with config.
- `reset(seed)`: Reset simulation.
- `step(action)`: Advance by control_dt, return obs, info.
- `get_cm_offset_magnitude()`: Current CM offset.

#### `HabitatEnv`

Gymnasium environment.

- `__init__(config)`: Initialize.
- `reset(seed)`: Reset, return obs, info.
- `step(action)`: Step, return obs, reward, terminated, truncated, info.

#### `ExperimentRecorder`

Database logger.

- `__enter__()`: Start recording.
- `record_episode(episode_num, steps_data, engine_info)`: Log episode.
- `__exit__()`: Flush.

### Key Functions

#### `run_training(config, recorder=None)`

Train SAC agent.

#### `evaluate_agent(model_path, config, n_episodes)`

Evaluate trained model.

#### `build_scenario_from_stochastic_config(stochastic_cfg, n_sectors, seed)`

Create stochastic disturbance scenario.

## Troubleshooting and FAQ

### Common Issues

**Torch installation fails on Windows**: Enable long paths as described in installation.

**Tests fail**: Ensure all extras are installed (`pip install -e ".[all]"`).

**RL training slow**: Reduce `n_envs` or timesteps.

**Database errors**: Ensure SQLite is available; use `:memory:` for testing.

### Performance Tips

- Use multiple environments (`n_envs > 1`) for parallel training.
- For evaluation, use deterministic policy.
- Simulation is CPU-bound; no GPU acceleration yet.

### Units and Conventions

- Length: meters
- Mass: kg
- Time: seconds
- Angles: radians (displayed in degrees where noted)
- Spin rate: rad/s (rpm in outputs)

### Extending the Codebase

See `IMPLEMENTATION_GUIDE.md` for adding features like vibration modes or new geometries.

For issues or contributions, refer to the project repository.
