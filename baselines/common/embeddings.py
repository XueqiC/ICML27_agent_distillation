"""Local-only encoders for the two registered non-Chinese embedding checkpoints."""
from pathlib import Path

from sed.seed_evidence import artifact_hashes

SPECS = {
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2", "hidden": 384, "layers": 6,
               "pooling": "attention_mask_mean", "max_length": 256},
    "bert": {"model": "google-bert/bert-base-uncased", "hidden": 768, "layers": 12,
             "pooling": "cls", "max_length": 512},
}


def identity(snapshot, kind):
    import json
    spec = SPECS[kind]
    config = json.loads((Path(snapshot) / "config.json").read_text())
    if (config.get("model_type") != "bert" or config.get("hidden_size") != spec["hidden"]
            or config.get("num_hidden_layers") != spec["layers"]):
        raise ValueError("Embedding snapshot architecture differs from registered model")
    return {**spec, "snapshot": str(Path(snapshot).resolve()), "artifact_sha256": artifact_hashes(snapshot)}


class LocalEncoder:
    def __init__(self, snapshot, kind):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.identity = identity(snapshot, kind)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False).cpu().eval()
        self.model.requires_grad_(False)

    def __call__(self, texts):
        torch = self.torch
        vectors = []
        for start in range(0, len(texts), 32):
            batch = self.tokenizer(texts[start:start+32], padding=True, truncation=True,
                max_length=self.identity["max_length"], return_tensors="pt")
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
                if self.identity["pooling"] == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                vectors.extend(torch.nn.functional.normalize(pooled, dim=-1).tolist())
        return vectors
