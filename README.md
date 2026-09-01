# Rate The Spire Path Rater

This project contains a second path rating model for Slay the Spire style maps.

The model rates candidate paths from the current node. It uses data that is known at the decision time:

- Character and ascension.
- Current act and floor.
- Current health and gold.
- Current deck, relic, and potion counts.
- The visible candidate path from the current node.
- A skill bucket from ascension, unless a dataset gives one.

The model does not use future card rewards, future gold, later act data, or future floor state arrays.

## Train

Prepare one decision row per candidate path. Use JSON, NDJSON, gzipped NDJSON, or CSV.

If the input only has the route that the player took, create chosen-path suffix rows:

```powershell
python .\path_rater_v2.py prepare --input .\data\runs.ndjson.gz --output .\data\decision_rows.ndjson
```

This mode can train a chosen-path rater. It cannot prove ranking quality for paths that the player did not take.

For STS1 run-history gzip files from the shared Drive corpus, keep the experiment in its own directories:

```powershell
python .\path_rater_v2.py prepare --mode sts1-run-history --input .\data\sts1_gdrive_sample\raw --output .\data\sts1_gdrive_sample\decisions.ndjson --max-files 3
python .\path_rater_v2.py train --input .\data\sts1_gdrive_sample\decisions.ndjson --model-out .\models\sts1_gdrive_sample\path_rater_neural.pkl --report-out .\reports\sts1_gdrive_sample\path_rater_neural_metrics.json --pos-weight auto
```

Use a separate output name when the local Drive sample is expanded:

```powershell
python .\path_rater_v2.py prepare --mode sts1-run-history --input .\data\sts1_gdrive_sample\raw --output .\data\sts1_gdrive_sample\decisions_12files_act_survival_sane.ndjson --label-mode act-survival
python .\path_rater_v2.py train-tabular-ensemble --input .\data\sts1_gdrive_sample\decisions_12files_act_survival_sane.ndjson --model-out .\models\sts1_gdrive_sample\path_rater_hgb_ensemble_12files_act_survival_sane.pkl --report-out .\reports\sts1_gdrive_sample\path_rater_hgb_ensemble_12files_act_survival_sane_metrics.json
```

This STS1 mode excludes daily, endless, and trial runs by default. It uses `path_per_floor`, current HP, current gold, cards picked up to the current floor, and relics obtained up to the current floor. Candidate paths stop at the current act boss. It does not use final `master_deck`, final `relics`, card reward options, final gold, or later acts as input features.

STS1 preparation also filters invalid current-state rows by default. Rows are skipped when HP is negative, max HP is missing or too large, HP is greater than max HP, gold is negative, or gold is over `5000`. Use `--include-invalid-state` only for a separate data-quality experiment.

Use `--label-mode act-survival` for a local target that predicts whether the player survives the current visible act path:

```powershell
python .\path_rater_v2.py prepare --mode sts1-run-history --input .\data\sts1_gdrive_sample\raw --output .\data\sts1_gdrive_sample\decisions_6files_act_survival_sane.ndjson --label-mode act-survival
```

For local Slay the Spire 2 `.run` files, use:

```powershell
python .\path_rater_v2.py prepare --mode sts2-run-history --input "$env:APPDATA\SlayTheSpire2\steam\<SteamID>\modded\profile1\saves\history" --output .\data\sts2_decisions.ndjson
```

This mode uses only the current floor state. It slices cards and relics by `floor_added_to_deck` and keeps candidate paths inside the current act.

By default, STS2 preparation excludes abandoned runs, modded-profile runs, and coop or multi-player runs. These runs are different populations. Use `--include-abandoned`, `--include-modded`, or `--include-coop` only for separate experiments.

Public STS2Runs snapshots also include cheated runs that are flagged by `_isCheated`. The prepare command excludes them by default. Use `--include-cheated` only for separate experiments.

For public STS2Runs gzip NDJSON snapshots, use:

```powershell
python .\path_rater_v2.py prepare --mode sts2-run-history --input .\data\runs-all-before-2026-06.json.gz --output .\data\sts2runs_public_standard_decisions.ndjson --game-mode standard
```

Use build or character filters for patch-aware experiments:

```powershell
python .\path_rater_v2.py prepare --mode sts2-run-history --input "$env:APPDATA\SlayTheSpire2\steam\<SteamID>\modded\profile1\saves\history" --output .\data\sts2_v01071_decisions.ndjson --build-id v0.107.1
```

```powershell
python .\path_rater_v2.py train --input .\data\decisions.ndjson --model-out .\models\path_rater_v2.pkl --report-out .\reports\path_rater_v2_metrics.json
```

Run the tabular comparison model:

```powershell
python .\path_rater_v2.py train-tabular --input .\data\decisions.ndjson --model-out .\models\path_rater_v2_hgb.pkl --report-out .\reports\path_rater_v2_hgb_metrics.json
```

Run the tabular ensemble model:

```powershell
python .\path_rater_v2.py train-tabular-ensemble --input .\data\decisions.ndjson --model-out .\models\path_rater_v2_hgb_ensemble.pkl --report-out .\reports\path_rater_v2_hgb_ensemble_metrics.json
```

Use `--configs` to pass comma-separated HGB settings as `learning_rate:max_iter:max_leaf_nodes:l2_regularization`.

Use `--blend-mode validation` to select ensemble weights from validation predictions. The default is `--blend-mode equal`.

Use `--sample-weight inverse-run` to give each training run equal total weight. Use `--sample-weight balanced` to balance class weight. Use `--sample-weight balanced-inverse-run` to combine both rules. The default is `--sample-weight none`.

Use `--threshold-mode slice` to learn validation-only thresholds for current-state buckets. The default is `--threshold-mode global`.

Training reports include split metadata. Use this to verify train, validation, and test row counts, run counts, class rates, and run-group overlap.

Analyze held-out performance by slice:

```powershell
python .\path_rater_v2.py analyze-slices --model .\models\path_rater_v2_hgb_ensemble.pkl --input .\data\decisions.ndjson --output .\reports\path_rater_v2_slice_analysis.json
```

The slice report includes overall held-out metrics, split metadata, per-slice metrics, and the lowest positive-F1 slices. Use `--min-rows` to suppress small buckets.

Neural training uses automatic positive-class weighting by default. Use `--pos-weight none` to disable it or pass a numeric weight.

Neural training uses CUDA automatically when a CUDA-enabled PyTorch build is installed. Use `--device cuda` to require CUDA and fail fast if CUDA is not available.

The neural path encoder uses learned position embeddings and attention pooling by default. Use `--no-position` or `--no-attention-pooling` for ablation tests.

The current feature set includes route-risk interactions for long paths, low HP, late-act position, visible elite density, visible monster density, and character-act context.

Use chronological evaluation when you need a forward-time test:

```powershell
python .\path_rater_v2.py train-tabular --input .\data\decisions.ndjson --split-strategy chronological
```

Required fields:

- `candidate_path`
- `victory` or `label`

Useful fields:

- `character`
- `ascension`
- `act`
- `floor`
- `current_hp`
- `max_hp`
- `current_gold`
- `deck`
- `relics`
- `potions`

## Score Paths

Create `state.json` for the current node and `candidates.json` with visible path candidates.

```powershell
python .\path_rater_v2.py score --model .\models\path_rater_v2.pkl --state .\state.json --candidates .\candidates.json
```

The highest `win_probability` is the recommended path.

For production STS2 use, enumerate candidates from `current_run.save` and its `saved_map`. Do not use future rewards or later act data.

```powershell
python .\path_rater_v2.py state-from-save --input .\current_run.save --output .\state.json
python .\path_rater_v2.py enumerate-map --input .\current_run.save --current-node "3,7" --output .\data\current_candidates.json
python .\path_rater_v2.py score-map --model .\models\path_rater_v2_hgb.pkl --state .\state.json --map .\current_run.save --current-node "3,7"
```

Evaluate where a player-chosen path ranks among all enumerated paths from an exported map:

```powershell
python .\path_rater_v2.py eval-map-choice --model .\models\path_rater_v2_hgb.pkl --state .\state.json --map .\current_run.save --current-node "3,7" --chosen-path "M-E-R-BOSS"
```

This reports the chosen path rank, percentile, probability gap to the top path, and all scored candidates. It needs a saved or exported map graph. A seed string by itself is not enough unless a trusted seed-to-map oracle is also available.

Evaluate many exported-map choices at once:

```powershell
python .\path_rater_v2.py eval-map-choices --model .\models\path_rater_v2_hgb.pkl --input .\data\map_choice_examples.ndjson --output .\reports\map_choice_eval.json
```

Each input row must include `state`, `map`, and `chosen_path`. `state` and `map` can be inline JSON values or paths relative to the input file. The batch report includes top-1 rate, top-3 rate, mean reciprocal rank, mean rank percentile, and mean probability gap to the top path.

## Audit

Audit a prepared dataset before training:

```powershell
python .\path_rater_v2.py audit --input .\data\sts2_vanilla_solo_decisions.ndjson --output .\reports\sts2_vanilla_solo_audit.json
```

The audit reports row counts, run counts, class balance, build mix, character mix, and whether the rows match the main vanilla solo population.

## Test

Run unit tests:

```powershell
python -m unittest discover -s tests
```

Run a synthetic smoke test:

```powershell
python .\path_rater_v2.py self-test
```

The smoke test proves that the training loop works. It does not prove real game F1.
