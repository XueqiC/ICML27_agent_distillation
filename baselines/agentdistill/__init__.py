"""AgentDistill: frozen student with a teacher-created MCP box."""

RECIPE = {
    "paper": "https://arxiv.org/html/2506.14728v1",
    "spec_sha256": "7d42a4e3d937c78b56f2045a8f42e4b96b55c2fc94237b250024cc0141ee42d8",
    "reference_commit": "83da61021941ae717f86ddf61e8b8e1d45ae0206",
    "reference_prompt_source_sha256": "e9bc2aedbbfb7bc7f949fffe50b9c6571b37b2e6253a5c45e988cdc6b0e7d575",
    "teacher": "gpt-6-luna",
    "runs_per_task": 3, "manager_turns": 40, "creations_per_turn": 1,
    "environment_actions": 40, "parameters": 3, "training_exposure": 0,
    "student_thinking": "off", "student_tokens": 1024, "evaluation_seed": 0,
    "label": "different interface", "box_steps": ["abstract", "cluster", "consolidate"],
    "sandbox": "bounded_python_ast_v1", "sandbox_fuel": 10000,
    "teacher_output_tokens": 32768, "box_output_tokens": 32768,
}


def teacher_cap(tasks):
    # 3 runs * 40 turns * (manager + creator), then at most one abstraction
    # and one consolidation per created tool, plus one global clustering call.
    return 3 * 40 * 4 * tasks + 1
