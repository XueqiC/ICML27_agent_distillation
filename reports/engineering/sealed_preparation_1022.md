# Sealed CPU preparation — 2026-10-22

Implemented on `wt/sealedprep`, starting at `7014900` (includes `fbd65bc`).
No commits, cluster actions, GPU execution, network access or teacher API calls.

## Behavior

- Collection and checked-support validation memoize successful results by resolved
  root, top-level file hashes, manifest/freeze digests and receipt-cache mode.
  Repeated calls return the same object; transitive collection anchors remain checked.
- Kang, AD-FTP and SmartAD CPU preparation retain the existing validation and
  encoding paths and publish exact encoded examples plus a seal under
  `<training-output>/prepared/<pipeline-identity-digest>/`.
- Pipeline training and trainer rank zero independently verify identity, recipe,
  config/code, every bound collection's top-level files, snapshot artifact hashes,
  tokenizer/template binding and example hashes. Missing or invalid seals refuse
  training and instruct the operator to run CPU preparation. Training reads the
  encoded examples without collection/episode scans or tokenizer encoding.
- Trainer provenance records the seal path and hashes; startup timing includes
  `seal_verification`. See [operating instructions](../../docs/HASH_RECEIPTS.md).

Published collections and their contents are immutable by protocol. GPU seal
verification deliberately does not detect episode-only changes that leave
collection anchors unchanged. Fresh CPU preparation performs the full evidence
audit; `SED_HASH_CACHE=0` also forces byte hashing rather than metadata receipts.

## Validation

Interpreter: `/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python`.
Every invocation used `CUDA_VISIBLE_DEVICES=''`; full-suite runs also set
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. The existing offline fixture blocks network
and OpenAI transport. All 111 test files ran, split over four local pytest
processes using `sorted(Path('tests').glob('test_*.py'))[index::4]`.

The full run reported **2,589 passed, 15 failed, 6 skipped** in 708 seconds.
After supplying missing pinned dependencies and updating pre-seal trainer test
fixtures, **13 affected failures passed** in a focused rerun (116 seconds).
One long collection-resume test also needed a rerun after a concurrent test-file
edit invalidated its intentionally strict code identity. The final combined
result is **2,602 passed, 6 skipped, 2 remaining failures**:

1. `test_two_process_gloo_matches_one_process_lora`: Gloo requires TCP sockets;
   the sandbox rejected socket creation with `Operation not permitted`. No
   network permission was requested and the test was not weakened or skipped.
2. `TestIndependentTheoryQuick::test_quick_has_no_failed_claims`: the independent,
   unchanged theory checker reports counterexample **C15** for equation (39)
   with response coverage and unaudited state-dependent provider law:
   observed gap `1`, claimed bound `0.029604143746015967`. This negative result
   was preserved; no theory or assertion was changed.

All **69** tests in `test_prepared_seal.py`, `test_validation_memo.py` and
`test_training_prepare.py` passed. Coverage includes round trips; missing seals;
tampered examples, collection/view/parent/support metadata, identity, code,
config, recipes, snapshot/tokenizer, tasks and split files; identical memoized
objects and validation counts; and synthetic prepare-to-train publication for
all three methods. The latter checks exact example equality and makes any
training-time collection validation, tokenizer load or encoding fail the test.
Separate existing CPU numerical tests retain their mask, update and provenance
assertions with prepared data supplied at the verification boundary.

The missing upstream files were made available temporarily through a read-only
use of the existing local `ICML27_agent_distillation/third_party` checkout.
LEAP matched pin `a3376d89a2eeee88f03555a4174a0bdf7faf8be6`; CoA and GAD's existing
source checks passed. The theory test's local interpreter path temporarily
pointed at the required interpreter above. These links and generated quick-test
figures were removed after validation; no repositories were copied or fetched.

Full logs/XML: `/tmp/icml27-sealedprep-suite-{0,1,2,3}.{log,xml}`.
Rerun: `/tmp/icml27-sealedprep-rerun.{log,xml}`.
C15 detail: `/tmp/icml27-sealedprep-dependencies.log`.

No HPG timing claim is made. The tests establish that the sealed training path
avoids episode validation/encoding; cluster wall-time improvement remains unmeasured.
