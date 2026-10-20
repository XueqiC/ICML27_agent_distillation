"""GAD, Ye et al. (2025), official LMOps/verl equations, paper-1 H0 path."""

RECIPE = {
    "spec_sha256": "7d42a4e3d937c78b56f2045a8f42e4b96b55c2fc94237b250024cc0141ee42d8",
    "lmops_commit": "6f53ef41c951de0962bbf27fe3f5819eabc8ce0e",
    "verl_commit": "707757b9202c466d986e529a4658434d0ac6a7ee",
    "adaptation": "full_parameter_text_fsdp2_verl_equations",
    "rounds": 3, "warmup_epochs": 1, "joint_epochs": 2,
    "discriminator_pre_steps": 10, "group_size": 8,
    "learning_rate": 1e-6, "original_batch": 256,
    "temperature": 0.8, "top_p": 1.0, "top_k": -1,
    "clip_ratio": 0.2, "dual_clip": 3.0, "kl_coef": 0.001,
    "kl_type": "low_var_kl", "advantage_epsilon": 1e-6,
    "weight_decay": 0.01, "gradient_clip": 0.2,
    "scheduler": "constant", "training_seed": 0,
    "loss_reduction": "global_response_token_mean",
    "max_response_length": 1536, "generation_cap": 1024,
    "max_length": 32768, "max_policy_turns": 40,
    "teacher_cap": "3 * 40 * support_unique_instances",
    "reference": "frozen_generator_after_warmup",
    "warmup": "10 discriminator-only updates, then one CE+BT epoch",
    "publication": "every generator optimizer update; no dev selection",
}
