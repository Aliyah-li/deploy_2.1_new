# Fixed-Height Simulation Entrypoints

Use these scripts for policies trained with a fixed action reference / fixed-height observation layout.

- `fft_simulation.sh`: residual FFT / self-supervised / multiscale FFT policy. Uses `POLICY_SUBDIR=fixed_height_fft` and `rl_deploy_fft`.
- `flat_simulation.sh`: fixed-height flat PPO base policy. Uses `POLICY_SUBDIR=fixed_height_flat` and `rl_deploy_fft`.

Expected policy files:

- `src/M20_sdk_deploy/policy/fixed_height_fft/policy.onnx`
- `src/M20_sdk_deploy/policy/fixed_height_flat/policy.onnx`
