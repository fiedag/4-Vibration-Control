Here is a complete reference for every parameter of the `habitat-train` script, grouped by concern:

---

## `habitat-train` — Complete Parameter Reference

The command is invoked as:
```bash
habitat-train [OPTIONS]
```
or equivalently:
```bash
python -m habitat_sim.scripts.train_agent [OPTIONS]
```

---

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | `None` | Path to a JSON `ExperimentConfig` file. If omitted, the built-in `reference_config()` is used (10 m cylinder, ~2 rpm). |
| `--timesteps` | `int` | `None` | **Override** `rl.total_timesteps` — total environment steps to train for. |
| `--n-envs` | `int` | `None` | **Override** `rl.n_envs` — number of parallel environments during training. |
| `--log-dir` | `str` | `None` | **Override** `rl.log_dir` — folder where models, checkpoints, and eval logs are saved. |
| `--seed` | `int` | `None` | Random seed for reproducibility (env resets, network init, etc.). |
| `--episode-duration` | `float` | `None` | Episode length in **seconds** (overrides `simulation.duration`). Default in config is 3600 s (1 hour); for fast training runs, use 60. |
| `--no-curriculum` | `flag` | `False` (curriculum ON) | Disables the 4-stage progressive difficulty ramp. When set, the agent trains at full difficulty from step 1. |
| `--db` | `str` | `None` | Path to a **SQLite database** file for telemetry/telemetry recording. If not provided, no DB is written. |
| `--experiment-name` | `str` | `"sac_run"` | Label stored in the database to identify this run among multiple experiments. |

---

### Config File Parameters (`ExperimentConfig` JSON)

The `--config` file controls everything not exposed directly on the CLI. Here are all sub-sections:

#### `habitat` — Physical structure
| Key | Default | Description |
|---|---|---|
| `shape` | `"cylinder"` | Geometry: `"cylinder"`, `"ring"`, or `"toroid"` |
| `radius` | `10.0` m | Outer radius (R for cylinder/ring; major radius for toroid) |
| `length` | `20.0` m | Axial length (cylinder/ring only) |
| `minor_radius` | `2.0` m | Toroid tube radius (unused for cylinder) |
| `wall_thickness` | `0.01` m | Shell wall thickness |
| `wall_density` | `2700.0` kg/m³ | Shell material density (aluminium default) |
| `end_plate_thickness` | `0.01` m | End cap thickness (cylinder only) |
| `end_plate_density` | `2700.0` kg/m³ | End cap density |

#### `sectors` — Interior discretisation
| Key | Default | Description |
|---|---|---|
| `n_angular` | `12` | Number of angular sectors (like slices of a pie) |
| `n_axial` | `3` | Number of axial stations along the length (use `1` for toroid) |

Total sectors = `n_angular × n_axial` = **36** by default.

#### `tanks` — Water ballast system
| Key | Default | Description |
|---|---|---|
| `n_tanks_per_station` | `12` | Tanks at each axial station |
| `n_stations` | `3` | Number of axial stations |
| `tank_capacity` | `100.0` kg | Max water per individual tank |
| `total_water_mass` | `1800.0` kg | Total water mass (36 tanks × 50 kg each by default) |
| `initial_distribution` | `"uniform"` | How water is distributed at episode start |
| `q_circ_max` | `5.0` kg/s | Max circumferential pump flow rate |
| `q_axial_max` | `1.0` kg/s | Max axial transfer flow rate |
| `k_axial` | `0.1` 1/s | Axial equalisation gain |

#### `motor` — Spin-up profile
| Key | Default | Description |
|---|---|---|
| `profile` | `"trapezoidal"` | Torque profile: `"constant"`, `"ramp"`, `"trapezoidal"`, `"s_curve"` |
| `max_torque` | `500.0` N·m | Peak motor torque |
| `ramp_time` | `60.0` s | Time to ramp from 0 to max torque |
| `hold_time` | `300.0` s | Time to hold at max torque (trapezoidal only) |
| `target_spin_rate` | `0.2094` rad/s | Target angular velocity (~2 rpm → ~0.44 g at 10 m) |

#### `simulation` — Integrator timing
| Key | Default | Description |
|---|---|---|
| `dt` | `0.01` s | Physics timestep (100 Hz) |
| `duration` | `3600.0` s | Episode duration (overridden by `--episode-duration`) |
| `control_dt` | `0.1` s | RL decision interval (10 Hz; agent acts every 10 physics steps) |
| `dynamics_level` | `1` | Fidelity level: `1`=rigid-body only, `2`=flex, `3`=full |

#### `rl` — SAC hyperparameters
| Key | Default | Description |
|---|---|---|
| `algorithm` | `"SAC"` | Algorithm identifier (currently only SAC is implemented) |
| `total_timesteps` | `500,000` | Total env steps to train (overridden by `--timesteps`) |
| `n_envs` | `4` | Parallel environments (overridden by `--n-envs`) |
| `learning_rate` | `3e-4` | Adam learning rate for actor and critic |
| `buffer_size` | `100,000` | Replay buffer capacity (in transitions) |
| `batch_size` | `256` | Mini-batch size per gradient update |
| `learning_starts` | `5,000` | Steps of random exploration before learning begins |
| `gamma` | `0.99` | Discount factor |
| `tau` | `0.005` | Soft update coefficient for target network |
| `ent_coef` | `"auto"` | Entropy regularisation: `"auto"` (learned) or a fixed float |
| `net_arch` | `[256, 256]` | MLP hidden layer sizes for actor and critic |
| `eval_freq` | `5,000` | Env steps between evaluation runs |
| `n_eval_episodes` | `5` | Episodes per evaluation cycle |
| `checkpoint_freq` | `25,000` | Steps between checkpoint saves |
| `log_dir` | `"./runs"` | Output directory (overridden by `--log-dir`) |
| `curriculum` | `true` | Progressive difficulty ramp (overridden by `--no-curriculum`) |

#### Curriculum stages (when `curriculum: true`)
| Stage | Timestep fraction | Imbalance mass |
|---|---|---|
| 0 | 0–25% | 0 kg (no disturbance) |
| 1 | 25–50% | 50 kg |
| 2 | 50–75% | 150 kg |
| 3 | 75–100% | 200 kg |

#### `stochastic` — Random disturbances (Phase 6)
| Key | Default | Description |
|---|---|---|
| `poisson_crew` | `false` | Enable random crew movement events |
| `n_crew` | `6` | Number of crew members |
| `mass_per_person` | `80.0` kg | Mass per crew member |
| `lambda_rate` | `0.01` /s | Mean sector transition rate |
| `transfer_duration` | `30.0` s | Smooth transition time per move |
| `micro_impacts` | `false` | Enable micro-impact disturbances |
| `impact_rate` | `0.001` /s | Mean impacts per second |
| `impact_mass_std` | `0.1` kg | Std of impact mass distribution |
| `impact_duration` | `1.0` s | Duration of each impact |

#### `seed`
| Key | Default | Description |
|---|---|---|
| `seed` | `42` | Global random seed for all stochastic elements (overridden by `--seed`) |

---

### Usage Examples

**1. Quickest possible run — defaults only:**
```bash
habitat-train
```

**2. Short smoke test (60 s episodes, 10k steps, no curriculum):**
```bash
habitat-train --timesteps 10000 --episode-duration 60 --no-curriculum --log-dir ./runs/smoke
```

**3. Full training with a custom config and DB logging:**
```bash
habitat-train --config configs/level1.json --log-dir ./runs/exp01 --db ./telemetry.db --experiment-name "level1_run1"
```

**4. Override parallelism and seed on top of a config file:**
```bash
habitat-train --config configs/level1.json --n-envs 8 --seed 123 --timesteps 1000000
```

**5. Minimal `configs/level1.json` example:**
```json
{
  "simulation": { "duration": 60, "dynamics_level": 1 },
  "rl": {
    "total_timesteps": 50000,
    "n_envs": 2,
    "learning_rate": 3e-4,
    "curriculum": true,
    "log_dir": "./runs/level1"
  },
  "seed": 42
}
```
All sections not listed fall back to their dataclass defaults.

---

### What gets saved in `log_dir`
| Path | Contents |
|---|---|
| `best_model/` | Best policy found during eval callbacks |
| `checkpoints/sac_ckpt_*` | Periodic snapshots every `checkpoint_freq` steps |
| `eval/` | Evaluation reward logs |
| `tb/` | TensorBoard logs (if `tensorboard` is installed) |
| `final_model.zip` | Final model after all timesteps complete |
