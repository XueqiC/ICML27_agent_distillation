"""Harness-Aware Distillation, arXiv:2610.02858v1 (paper implementation)."""

RECIPE = {
    "spec_sha256": "7d42a4e3d937c78b56f2045a8f42e4b96b55c2fc94237b250024cc0141ee42d8",
    "paper": "https://arxiv.org/html/2610.02858v1", "training_seed": 0,
    "adaptation": "full_parameter_text_fsdp2", "learning_rate": 1e-5,
    "scheduler": "cosine", "warmup_steps": 15, "weight_decay": 0.01,
    "gradient_clip": 1.0, "original_batch": 60, "updates": 250,
    "beta": 0.5, "rho": 0.5, "teacher_cap": 30000, "tasks_per_round": 16,
    "max_policy_versions": 2, "policy_version_unit": "published_rollout_round",
    "rollout_temperature": 1.0, "evaluation_temperature": 0.0,
    "packing": "turn", "max_length": 32768, "step_cap": 40,
    "stall_window": 5, "failed_attempts": 2, "budget_warning": 11,
}
