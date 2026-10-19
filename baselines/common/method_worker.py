"""CPU embedding work in the training interpreter; never allocates a CUDA device."""
import argparse
import os

from baselines.common.embeddings import LocalEncoder
from baselines.common.method_data import read
from sed.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=("wkm", "steca"))
    for key in ("config", "verified", "embedding-snapshot", "data", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--reflections")
    p.add_argument("--sources", nargs="*", default=[])
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = load_config(a.config)
    encode = LocalEncoder(a.embedding_snapshot, "minilm" if a.method == "wkm" else "bert")
    if a.method == "wkm":
        from baselines.wkm.retrieval import build
        build(a.data, a.output, config, a.verified, encode)
    else:
        from baselines.steca.data import merge
        merge(read(a.data), a.reflections, a.output, config, a.verified, [a.data, *a.sources], encode)


if __name__ == "__main__":
    main()
