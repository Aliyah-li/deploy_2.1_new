# Sim2Sim Deploy Config — Strong Base + FFT Residual

Policy-to-deployment bridge document. Keep synced with the checkpoint used in production.

> **Canonical ordering rule:** never infer observation indices from semantic
> grouping. Use the exported training `env.yaml`. `base_height_command` is
> appended in `__post_init__`, therefore it is `obs[57]`, not `obs[9]`.

---

## 1. Control Parameters

| Parameter | Value | Notes |
|---|---|---|
| Control rate | **50 Hz** | `step_dt = 0.02 s` |
| Physics dt | 0.005 s | 4× decimation |
| Episode length | 20 s | training only |

---

## 2. Command Interface

### 2.1 Base Velocity Command (3 dims)

| Field | Range | Unit |
|---|---|---|
| `vx` (lin_vel_x) | **0.0 → 1.75** | m/s |
| `vy` (lin_vel_y) | **0.0 → 0.0** | m/s |
| `yaw_cmd` (ang_vel_z) | **-0.785 → +0.785** | rad/s |

- Training samples a target heading internally, but `generated_commands` exposes
  the resulting `ang_vel_z` command as the third policy input. Deployment must
  provide the yaw-rate command in this slot, not an absolute yaw angle.
- `rel_standing_envs = 0.05` — 5% of envs get `vx=0` (standstill).
- Resample period: **10 s**.

### 2.2 Base Height Command (1 dim)

| Field | Range | Unit |
|---|---|---|
| `target_height` | **0.32 → 0.425** | m |

- Nominal height: **0.40 m**.
- Resample period: **10 s**.

---

## 3. Action Space

### 3.1 Joint Position Actions (12 dims) — Leg Joints

Order (preserve_order=True):
```
[0]  fl_hipx_joint      (steering, left front)
[1]  fl_hipy_joint      (hip abduction, left front)
[2]  fl_knee_joint      (knee, left front)
[3]  fr_hipx_joint      (steering, right front)
[4]  fr_hipy_joint      (hip abduction, right front)
[5]  fr_knee_joint      (knee, right front)
[6]  hl_hipx_joint      (steering, left rear)
[7]  hl_hipy_joint      (hip abduction, left rear)
[8]  hl_knee_joint      (knee, left rear)
[9]  hr_hipx_joint      (steering, right rear)
[10] hr_hipy_joint      (hip abduction, right rear)
[11] hr_knee_joint      (knee, right rear)
```

| Setting | Value |
|---|---|
| Type | Position control |
| Scale | `{".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25}` |
| Offset | 0.0 (use_default_offset=True) |
| Clip | [-100, 100] |

**Effective range**: action ∈ [-1, 1] → joint offset ∈ [-scale, +scale]:
- hipx: ±0.125 rad (±7.2°)
- hipy/knee: ±0.25 rad (±14.3°)

### 3.2 Joint Velocity Actions (4 dims) — Wheel Joints

Order:
```
[12] fl_wheel_joint
[13] fr_wheel_joint
[14] hl_wheel_joint
[15] hr_wheel_joint
```

| Setting | Value |
|---|---|
| Type | Velocity control |
| Scale | **5.0** |
| Clip | [-100, 100] |

---

## 4. Default Joint Pose (URDF)

```
hipx (all):      0.0 rad
hipy (fl, fr):  -0.6 rad
hipy (hl, hr):  +0.6 rad
knee (fl, fr):  +1.0 rad
knee (hl, hr):  -1.0 rad
wheel (all):     0.0 rad
```

Actions are **offsets from this default** (`use_default_offset=True`).

---

## 5. StandUp IK Reference (Not the Policy Action Offset)

The stand-up controller uses M20 leg IK (`standup_state.hpp`,
`thigh=0.25m, shank=0.25m`) to reach its hand-off pose:

| h (m) | hip_y (rad) | hip_y (°) | knee (rad) | knee (°) |
|---|---|---|---|---|
| 0.320 | -0.876 | -50.2° | 1.753 | 100.5° |
| 0.380 | -0.707 | -40.5° | 1.415 | 81.1° |
| 0.400 | -0.644 | -36.9° | 1.287 | 73.7° |
| 0.425 | -0.555 | -31.8° | 1.110 | 63.6° |

**Sign convention** (deploy-level, matches `standup_state.hpp`):
- FL/FR (front): hipy = +θ, knee = +θ_k
- HL/HR (rear):  hipy = -θ, knee = -θ_k
- hipx: 0 (steering locked straight in base policy)

This table is only a stand-up/reference diagnostic. The RL policy action and
`joint_pos_rel` observation always use the fixed URDF defaults in section 4.
Do not add the IK pose to policy actions and do not subtract it in observations.

---

## 6. Observation Structure (Policy — 58 dims)

| Index | Name | Dims | Scale |
|---|---|---|---|
| 0:3 | base_ang_vel | 3 | 0.25 |
| 3:6 | projected_gravity | 3 | 1.0 |
| 6:9 | velocity_commands (vx, vy, heading) | 3 | 1.0 |
| 9:25 | joint_pos (16 joints, wheel positions zeroed) | 16 | 1.0 |
| 25:41 | joint_vel (16 all joints) | 16 | 0.05 |
| 41:57 | actions (prev 16) | 16 | 1.0 |
| 57:58 | base_height_command | 1 | 1.0 |

*base_lin_vel and height_scan disabled.*

---

## 7. Observation Normalization

- **Actor**: obs_normalization = **False** (no EmpiricalNormalization).
- If deploying ONNX exported model: input = raw observation values in the ranges above.

---

## 8. Network Architecture

| Component | Spec |
|---|---|
| Actor type | `MLPModel` (standard RSL-RL MLP) |
| Hidden dims | [512, 256, 128] |
| Activation | ELU |
| Output | 16 dims (Gaussian mean) |
| Input | 58 dims (policy obs) |
| Distribution | Gaussian, init_std=1.0, std_type=log |

---

## 9. Checkpoint Paths

| Role | Path |
|---|---|
| Strong base (flat+slope) | `.../strong_ppo_base/..._shv2/model_999.pt` |
| FFT residual (full terrain) | `.../active_perceptive_residual_fft_terrain/..._n2/model_XXXX.pt` |

---

## 10. Deployment Checklist

- [ ] Joint order matches section 3.1 exactly (preserve_order)
- [ ] Action scale matches: hipx=0.125, hipy/knee=0.25, wheel=5.0
- [ ] use_default_offset mode: apply action as offset from URDF default
- [ ] Command interface: vx + heading + height, as section 2
- [ ] vx=0 triggers standstill (joints → default)
- [ ] Height is stored only at observation index 57; do not insert it at index 9
- [ ] Section 5 IK is used only for stand-up/handoff, not as the RL action offset
- [ ] Obs normalization: off (raw values)
- [ ] 50 Hz control rate
