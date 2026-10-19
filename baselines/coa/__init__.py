"""Chain-of-Agents, Li et al. (2025), ALFWorld/H0 port."""
PIN = "9a6594d4661fa994f189074ecfdfd259eea87c6d"
RECIPE = {
    "official_revision": PIN,
    "spec_sha256": "d6cc35f3222ee234cc9c970fe2392174365bf48e23ed51ba7e6c1ed91cee4f76",
    "training_seed": 0, "adaptation": "full_parameter_text_fsdp2", "max_length": 32768,
    "teacher_runs": 3, "teacher_eval_runs": 1, "max_calls_per_step": 8,
    "teacher_output_tokens": 1024, "min_interactions": 5,
    "nonreflection_keep_probability": .5, "error_correction_copies": 2,
    "sft": {"learning_rate": 1.4e-5, "epochs": 4, "original_batch": 128,
            "scheduler": "linear", "warmup_ratio": .1, "weight_decay": 0.0},
    "rl": {"learning_rate": 1e-6, "epochs": 2, "max_updates": 60,
           "original_batch": 64, "generation_batch": 128, "mini_batch": 64,
           "samples": 8, "temperature": 1.0, "warmup_steps": 3,
           "weight_decay": .1, "clip_low": .2, "clip_high": .28, "clip_c": 10.,
           "ppo_epochs": 1, "loss_aggregation": "token-mean", "kl_coef": 0.0},
}
