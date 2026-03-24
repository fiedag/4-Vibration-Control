# Habitat Vibration Control Simulator — Project Instructions

You are an expert assistant for a **rotating space habitat dynamics simulation and reinforcement learning control system** called `habitat_sim`. This Python codebase simulates a large rotating structure (cylinder, ring, or toroid) in microgravity that generates artificial gravity through spin. As crew and cargo move, mass imbalances create precession, nutation, and vibration. A SAC (Soft Actor-Critic) neural network is trained to suppress these imbalances by pumping water between rim tanks in real time.

## Project Status

- **113 tests passing** across 6 implementation phases
- ~4,177 lines of production code, ~2,179 lines of test code
- Level 1 (rigid body) dynamics fully implemented; Levels 2–3 (flexible body) planned

## Technology Stack

- **Python 3.10+** with numpy, scipy
- **RL:** Gymnasium >= 0.29, Stable Baselines 3 >= 2.0, PyTorch >= 2.0
- **Database:** SQLAlchemy 2.0 with SQLite for telemetry
- **Visualization:** matplotlib, plotly
- **Build:** pyproject.toml with optional extras (`rl`, `db`, `viz`, `config`, `dev`, `all`)

## Architecture Overview

```
habitat_sim/
├── config.py              — Configuration dataclasses (all phases)
├── core/                  — Physics fundamentals
│   ├── quaternion.py      — Quaternion math (multiply, normalize, derivatives)
│   ├── inertia.py         — Inertia tensor computation from geometry & point masses
│   └── integrator.py      — RK4 integrator with quaternion normalization
├── geometry/              — Habitat shape models
│   ├── base.py            — Abstract HabitatGeometry interface
│   ├── cylinder.py        — Cylinder with end plates + factory
│   └── toroid.py          — Thin-walled toroidal shell (Phase 6)
├── dynamics/              — Physics models
│   ├── base.py            — Abstract DynamicsModel interface
│   └── rigid_body.py      — Level 1: Euler equations + tank ODEs
├── actuators/             — Control mechanisms
│   ├── motor.py           — Spin motor with torque profiles
│   └── tank_system.py     — 36 rim tanks + hybrid 3-manifold plumbing
├── sensors/               — Observation interface
│   ├── strain_gauge.py    — Floor force sensor per sector (with Gaussian noise)
│   └── sensor_suite.py    — Bundles sensors → 75-D observation vector
├── disturbances/          — Mass scheduling & perturbations
│   ├── mass_schedule.py   — Prescribed crew/cargo movement events
│   ├── scenario.py        — Combines multiple disturbance sources
│   └── stochastic.py      — Poisson crew movement + micro-impacts (Phase 6)
├── simulation/            — Core engine
│   ├── state.py           — SimState: 46-D state vector with named views
│   ├── engine.py          — SimulationEngine: orchestrates all components
│   └── monitors.py        — Conservation checks (angular momentum, water mass, energy)
├── environment/           — RL interface
│   └── habitat_env.py     — Gymnasium environment wrapping SimulationEngine
├── control/               — RL training (Phase 4)
│   ├── sac_agent.py       — SAC + MLP policy, vectorized env helpers
│   └── training.py        — Training loop, curriculum learning, evaluation
├── database/              — Telemetry (Phase 5)
│   ├── schema.py          — SQLAlchemy 2.0 ORM
│   ├── recorder.py        — ExperimentRecorder context manager + SB3 callback
│   └── queries.py         — Analysis helpers
└── scripts/               — CLI tools
    ├── train_agent.py     — `habitat-train`
    ├── run_simulation.py  — `habitat-run`
    └── analyse_experiment.py — `habitat-analyse`
```

## Key Physical Parameters

| Parameter | Value |
|-----------|-------|
| State dimension | 46 (quaternion 4 + ω 3 + tanks 36 + manifolds 3) |
| Observation dimension | 75 (strain 36 + tank levels 36 + manifold levels 3) |
| Action dimension | 36 (valve commands per tank) |
| Physics timestep | 0.01 s (100 Hz) |
| Control timestep | 0.1 s (10 Hz) |
| Sectors | 36 (12 angular × 3 axial) |
| Tank capacity | 100 kg each |
| Target spin rate | 0.2094 rad/s (~2 rpm) |

## Physics Core

- **Coordinate frames:** Inertial frame (I) with Z_I as nominal spin axis; Body frame (B) rotating with habitat
- **Quaternion kinematics:** `dq/dt = ½ Ω(ω)·q` — avoids gimbal lock
- **Euler's equation:** `dL/dt = τ_motor + τ_gravity + τ_actuator`, solved as `dω/dt = I⁻¹(L - ω × I·ω)`
- **Integration:** RK4 with quaternion renormalization at each step
- **Tank dynamics:** circumferential flow ≤ 5 kg/s per valve; passive axial manifold mixing with gain 0.1/s

## RL Training Pipeline

- **Algorithm:** SAC with MLP [256, 256] policy
- **Curriculum learning** (4 stages): gradually increases mass imbalance from 0 kg → 200 kg
- **Reward:** combines vibration suppression, pump energy cost, command smoothness, reserve balance
- **Database telemetry:** experiments → episodes → timesteps recorded via SQLAlchemy ORM

## Testing

Tests are organized by implementation phase:
- `test_phase1.py` — Core physics (quaternion, inertia, integrator) — 24 tests
- `test_phase2.py` — Disturbances (mass schedules, scenarios) — 22 tests
- `test_phase3.py` — Sensors & Gymnasium env — 27 tests
- `test_phase4.py` — RL training (SAC, curriculum, evaluation) — 12 tests
- `test_phase5.py` — Database (SQLite ORM, recorder) — 12 tests
- `test_phase6.py` — Stochastic disturbances + toroid geometry — 18 tests

Run all tests with: `pytest tests/ -v`

## How to Help

When assisting with this project:

1. **Respect the modular architecture** — each subpackage has clear responsibilities. Don't mix concerns.
2. **Follow existing patterns** — new dynamics models inherit from `DynamicsModel`, new geometries from `HabitatGeometry`, etc.
3. **Configuration-driven design** — all parameters live in `config.py` dataclasses. Don't hardcode values.
4. **Test thoroughly** — maintain the phase-based test organization. Each new feature needs corresponding tests.
5. **Conservation laws matter** — water mass and angular momentum must be conserved. The monitors in `simulation/monitors.py` check this.
6. **Physics accuracy** — quaternion math must maintain unit norm. RK4 integration preserves accuracy. Euler equations handle variable inertia from water movement.
7. **When suggesting code changes**, always consider the impact on the 46-D state vector and 75-D observation space.
8. **Entry points:** `habitat-train` and `habitat-run` are the main CLI commands.
