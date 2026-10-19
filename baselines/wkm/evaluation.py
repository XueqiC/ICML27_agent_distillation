"""Fail-closed dual-LoRA gate for the existing shared evaluation runner."""
from baselines.common.method_data import read
from sed.storage import digest
from tools.check_lora_serving import validate_report


def validate_dual(config, spec):
    server = config.get("server_provenance", {})
    if server.get("config", {}).get("server", {}).get("batch_invariant") is not True:
        raise ValueError("WKM requires batch-invariant serving")
    if (spec["model"] == spec.get("knowledge_model") or spec["model"] == config["models"]["student"]["requested_repo"]
            or spec.get("knowledge_model") == config["models"]["student"]["requested_repo"]):
        raise ValueError("WKM requires two distinct adapters on the configured backbone")
    for field, model in (("lora_equivalence", spec["model"]),
                         ("knowledge_lora_equivalence", spec.get("knowledge_model"))):
        if not config.get(field) or not model:
            raise ValueError("WKM requires passing reports for BOTH LoRAs")
        validate_report(config[field], server, model, spec["endpoint"])
        adapter = config[field]["identity"]["adapter"]
        if (adapter["base_model"] != config["models"]["student"]["requested_repo"]
                or adapter["revision"] != config["models"]["student"]["revision"]):
            raise ValueError("WKM adapters must share the configured Gemma backbone")
    payload = read(spec["knowledge_base"])
    checksum = payload.pop("sha256")
    if digest(payload) != checksum:
        raise ValueError("WKM knowledge base changed")
