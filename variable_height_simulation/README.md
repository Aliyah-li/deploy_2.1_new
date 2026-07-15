# Variable-Height Simulation Entrypoints

Use these scripts for policies trained with height command appended to the observation.

- `fft_simulation.sh`: variable-height FFT policy. Uses `POLICY_SUBDIR=variable_height_fft` and `rl_deploy_fft2`.
- `flat_simulation.sh`: variable-height flat PPO / strong2 policy. Uses `STRONG2_POLICY_SUBDIR=variable_height_flat` and `rl_deploy_strong2`.

Expected policy files:

- `src/M20_sdk_deploy/policy/variable_height_fft/policy.onnx`
- `src/M20_sdk_deploy/policy/variable_height_flat/policy.onnx`
