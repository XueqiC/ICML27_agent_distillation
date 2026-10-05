# Official baseline source pins

Transcribed from [the local port specification](../reports/baseline_port_specs_1004.md),
2026-10-04. These are source-code references, not permission to load any repo's
models or datasets. ETO is implemented in Stage 3a; AgentGym BC/AgentEvol and
Kang non-FTP SFT + SAG are implemented in Stage 3b; IPR and seeded SAG requests
are implemented in Stage 3c; LEAP with privileged training teacher and its
0%-privileged ablation are implemented in Stage 3d. No upstream files are modified.

| Method | Official repository | Commit |
| --- | --- | --- |
| ETO | https://github.com/Yifan-Song793/ETO | `a2fc5da38f8d00cfaf3f9b6370d586eebaf72904` |
| [IPR](ipr/DEVIATIONS.md) | https://github.com/WeiminXiong/IPR | `93335a3a818b46c1d0e081cd76288cf1cbf7dee4` |
| [AgentGym BC / AgentEvol](agentgym/DEVIATIONS.md) | https://github.com/WooooDyy/AgentGym | `3ef9235d23e68e7c2920c5422ad957dc8ced5c6c` |
| WKM | https://github.com/zjunlp/WKM | `f1fe28fc189c59801a98cabda01188bc9b64eae6` |
| [LEAP / 0% privileged ablation](leap/DEVIATIONS.md) | https://github.com/sanjibanc/leap_llm | `a3376d89a2eeee88f03555a4174a0bdf7faf8be6` |
| [Kang SFT + SAG (no FTP)](kang/DEVIATIONS.md) | https://github.com/Nardien/agent-distillation | `8884b80ea3d22e53a2e1e4b9fd324600d13e0430` |

`eto/official_loss.py` contains exactly two methods from ETO's FastChat/TRL fork:
`dpo_trainer.py:737-839`. Their method bodies, decorators, signatures and comments
are unchanged. The Apache-2.0 DPO/HuggingFace notice is retained; only imports
and a dependency-free containing class differ. Tests compare ASTs and execute
the original methods independently without importing the old trainer stack.

Production target: HPG torch **2.11.0+cu128**, transformers **5.14.1**, peft
**0.20.0**, vLLM **0.27.1**, as recorded in `configs/hpg.env` / the specification.
The local CPU verification runtime is torch **2.13.0**, transformers **5.14.1**;
it does not certify the HPG installation. Every future stage captures actual
installed versions and source/data/checkpoint hashes.
