  High-speed policy routing:

  ┌─────────────────┬─────────────────────────────────┐
  │  Direction key  │           Policy used           │
  ├─────────────────┼─────────────────────────────────┤
  │ 1 (x)           │ high_speed/x_rash/policy.onnx   │
  ├─────────────────┼─────────────────────────────────┤
  │ 2 (y)           │ high_speed/y_jump/policy.onnx   │
  ├─────────────────┼─────────────────────────────────┤
  │ 3 (diagonal)    │ high_speed/diagonal/policy.onnx │
  ├─────────────────┼─────────────────────────────────┤
  │ Low speed (any) │ low_speed/policy.onnx           │
  └─────────────────┴─────────────────────────────────┘

  mixed_terrains_history_vae deployment contract:

  - Exported ONNX input names: obs, obs_history
  - obs dim: 58
  - obs_history dim: 610 (10 flattened frames x 61 values)
  - Actor: HistoryVaeActor, MLP [512, 256, 128], ELU
  - VAE: latent dim 8, encoder/decoder [128, 64], policy input = mean
  - Source task config is mirrored in config/mixed_terrains_history_vae_config.json
