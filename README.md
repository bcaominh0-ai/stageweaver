# StageWeaver

This repository has been reset to the current StageWeaver mainline.

Active comparison:

1. `memento_text`
2. `stageweaver`

Canonical runner:

```bash
python client/stageweaver_runner.py --memory_mode memento_text
```

StageWeaver online evaluation requires the local direct backend, an append-aligned composer checkpoint, and `--stage_mode both`.

Current semantic workflow stages:

1. `collect_seen_train_traces`
2. `collect_seen_dev_traces`
3. `build_stageweaver_bank`
4. `train_append_sft`
5. `eval_seen_dev_gate`
6. `eval_ood_test`

Recommended current output directories:

- `result/stageweaver/current/tracebank_seen_train/`
- `result/stageweaver/current/tracebank_seen_dev/`
- `result/stageweaver/current/stage_bank/`
- `result/stageweaver/current/append_sft/`
- `result/stageweaver/current/eval_seen_dev/`
- `result/stageweaver/current/eval_ood_test/`

Current docs:

- `docs/CURRENT_PROTOCOL.md`
- `docs/CLEANUP_MANIFEST.md`
- `docs/ARCHIVED_RESULTS_SUMMARY.md`
