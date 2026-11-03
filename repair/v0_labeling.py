"""Local tokenizer resolution and audited, exact-request V0 label reuse."""
import json
from pathlib import Path

from repair.v0_common import check_inputs, completed, read
from sed.storage import canonical_bytes, digest, file_hash


class LabelSetupError(ValueError):
    """Safe, actionable setup errors that may be printed at the CLI boundary."""


def label_tokenizer(args, config, inputs):
    from tools.build_v2_dataset import local_tokenizer
    if getattr(args, 'byte_token_cap', False):
        if getattr(args, 'snapshot', None):
            raise LabelSetupError('--byte-token-cap cannot be combined with --snapshot')
        from repair.supervision import ByteBoundTokenizer
        return ByteBoundTokenizer(), dict(type='utf8_bytes', test_only=True)
    try:
        tokenizer, identity = local_tokenizer(args, {'config': config}, inputs)
    except ValueError:
        raise LabelSetupError('V0 label requires a valid pinned local tokenizer.json; check --snapshot') from None
    if tokenizer is None:
        raise LabelSetupError('V0 label requires local tokenizer.json; supply --snapshot (no byte fallback)')
    return tokenizer, identity


class LabelReuseSource:
    """Index completed raw outputs, never old validation decisions or usage.

    A durable requests/*.json receipt implies completion. Transport receipts
    additionally cover an interruption after completion but before that write.
    Equal requests use the first durable receipt in sorted path order; transport
    completions are a fallback. Keys are the actual canonical request bytes.
    """
    def __init__(self, directory, inputs, config_sha256):
        self.directory = Path(directory).resolve()
        plan = read(self.directory / 'plan.json')
        identity = plan['identity']
        if identity.get('command') != 'label' or plan['run_sha256'] != digest(identity):
            raise LabelSetupError('V0 label reuse requires an intact label phase plan')
        # Config files may move between worktrees; their bytes must still match.
        # Legacy plans bind one config file alongside manifest/attempt/review
        # inputs. New plans name those scientific inputs separately.
        source_inputs = identity.get('label_source_inputs')
        source_config = identity.get('label_config_sha256')
        if source_inputs is None:
            source_inputs = dict(identity['inputs'])
            extra = set(source_inputs) - set(inputs)
            if len(extra) == 1:
                source_config = source_inputs.pop(extra.pop())
        if source_inputs != inputs or source_config != config_sha256:
            raise LabelSetupError('V0 label reuse source inputs differ (manifest, attempts, reviews or config)')
        if not completed(self.directory):
            check_inputs(identity['inputs'])
        self.entries = {}
        for path in sorted(self.directory.glob('episodes/*/requests/*.json')):
            self._add(path, read(path))
        for path in sorted(self.directory.glob('episodes/*/teacher/requests.jsonl')):
            sha256 = file_hash(path)
            for line, text in enumerate(path.read_text().splitlines(), 1):
                row = json.loads(text)
                if row.get('status') == 'completed':
                    self._add(path, row, sha256=sha256, line=line)

    def _add(self, path, row, *, sha256=None, line=None):
        if not isinstance(row.get('request'), dict) or not isinstance(row.get('raw_output'), str):
            return
        receipt = dict(raw_output=row['raw_output'], reused=True,
                       source_path=str(path.resolve()), source_sha256=sha256 or file_hash(path))
        if line is not None:
            receipt['source_line'] = line
        self.entries.setdefault(canonical_bytes(row['request']), receipt)

    def find(self, request):
        return self.entries.get(canonical_bytes(request))
