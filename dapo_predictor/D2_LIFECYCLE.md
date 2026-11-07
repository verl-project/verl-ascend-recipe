# D2 training and activation

## Default: online CPU fitting

The default backend starts without a prediction checkpoint. Before each rollout, the actor extracts the configured hidden taps for one row per prompt group. The driver pairs those features with this rollout's sibling lengths and cap-censor flags before reward filtering or actor updates. Answers from the current rollout cannot affect the forecasts already used for it.

A single CPU subprocess fits immutable replay snapshots. The replay retains up to 512 recent prompt observations by default, each with its sibling lengths. The first fit needs 32 observations. The head is a gated fusion of projected taps followed by mu/sigma prediction for standardized log(1 + response_tokens). Exact observations use Gaussian density; censored observations use the log survival function. PCA projection and label normalization are fitted once; subsequent candidates retain those transforms and warm-start head and AdamW state. Each job performs 50 minibatch updates by default.

RL never waits for a pending fit. A busy fitter coalesces additional observations into its next snapshot instead of queueing multiple jobs. Before a new rollout, the driver polls for a completed candidate, validates its numeric/schema state, and loads the same hash-bound checkpoint onto every actor rank. Candidate loading finishes before any rank scores the batch. A fit failure leaves the previous head in place, or FCFS if no head has been published. An all-rank load failure aborts to prevent mixed heads.

EWS remains FCFS until all three conditions hold: ten complete RL dataset passes, 2560 observed rollout responses, and a published head. These gates are configurable in epochs and sample count; there is no step-based activation. Head training continues during and after warm-up. There is no error-threshold retraining rule or automatic D1/D2 switch. CPU fitting can overlap NPU work, but hidden extraction, transfer and candidate publication still have costs that must be measured.

Pre-response forecasts are evaluated against future completed outputs before adding their labels to training. Metrics include prompt-level Spearman of scheduling risk vs observed mean length, token MAE on fully uncensored prompt groups, p90 coverage lower/upper bounds and censored fraction. Constant forecasts or fewer than two exact prompts mark correlation undefined rather than manufacturing a score. Training loss is a separate metric. Repeated responses from the same prompt do not establish cross-prompt generalization.

Online checkpoints save the bounded replay, fixed feature schema/transforms, head, optimizer, published version, data cutoff and observation counters. An unfinished background job is not a published head; restored pending observations can be fitted again. Resume validates the backend, configuration, activation gates and replay/head counters before loading. Saving only an external weight path is insufficient for online resume.

For a short run ending at or before the warm-up boundary, the code warns that scheduling remains FCFS. A 20-epoch run leaves ten epochs after the default boundary, but does not guarantee that this amortizes overhead for every dataset size or model. Use the public launcher and device workload in [README.md](README.md).

## Explicit compatibility: external frozen D2

This section applies when `d2_online.enable=false`, for example by supplying `D2_CANDIDATE_REPORT` to the launcher. It does not describe default online training.

D2 uses an externally trained multi-tap LogNormal checkpoint. The recipe loads and validates that checkpoint; it does not collect a D2 training dataset or fit the head. Configure the shared candidate-report path before launch. All actor ranks must be able to read the immutable weight file named by the report.

## Candidate checkpoint

The checkpoint is a torch.save mapping with feature_keys (layer_<index> taps), projection_dim, projection_bank, standardized_sigma_floor, head_state_dict, log_length_center and log_length_scale. The projection bank contains center, components and scale tensors per tap. The head predicts a Gaussian over standardized log(1 + response_tokens); inference reverses the standardization before computing token statistics. Use the same actor, tokenizer, chat template and tap schema when collecting features and deploying the checkpoint.

The external collector/trainer must produce held-out, pre-response forecasts in the JSONL format below. A JSONL of forecasts alone does not train a head. See [README](README.md#predictor-configuration) for the D2 launch options.

## Fitting and validation

`activation.min_epoch=10` means **ten completed RL dataset passes**, so the earliest admission is before pass 11. It is not a cap on predictor optimizer steps, nor proof that ten passes suffice for every workload. At a boundary without a usable report the trainer continues FCFS; it does not block waiting for an external fit. Publish the candidate before a boundary or it will be considered five passes later. While inactive, collect additional data externally and refit/revalidate every five passes as needed. Once admitted, only monitor every five passes; do not refit routinely.

Use a fixed prompt population with held-out *future outputs*, not necessarily unseen prompt identities. For example, fit a candidate on passes 1–9 and validate on pass 10; if it fails, fit the next candidate through pass 14 and validate on pass 15. If fitting includes pass 10, that candidate must instead be validated on later outputs: its fitting rows cannot also certify admission. Feature projection/normalization/head fitting must all respect the cutoff. Collect validation forecasts before seeing their responses, using the matching actor/model/tokenizer/tap schema; a post-hoc feature readout is not online validation. These provenance obligations remain with the external pipeline; this utility checks the declared temporal boundary and file hashes, not the truth of those declarations.

Write one JSONL row per held-out response (siblings share the prompt identity):

```json
{"prompt_id":"stable-tokenized-prompt-hash","epoch":10,"observed":1200,"expected":1300,"median":1100,"p90":2400,"censored":false}
```

All epoch numbers in rows are one-based. Mark cap-truncated responses `censored: true`, including ambiguous EOS-at-cap cases conservatively. Actual lengths alone cannot distinguish those cases. Publish with:

```bash
python -m recipe.dapo_predictor.validate_d2 \
  --checkpoint /absolute/shared/path/d2-candidate-v1.pt \
  --rows /absolute/path/heldout-pass10.jsonl \
  --trained-through-epoch 9 --validation-epoch 10 \
  --output /absolute/shared/path/d2-candidate.json
```

The utility streams the input rows and atomically replaces the report; it rejects an output path that resolves to the checkpoint or input rows. The candidate weight file itself must be immutable and visible at the same absolute path on every actor rank. The trainer recomputes metrics from the report's grouped sufficient statistics, checks the checkpoint SHA-256 and matching `risk_weight`, and rejects stale/future/overlapping validation. It never trusts a supplied `passed` flag. Publishing is a trusted local handoff, not a cryptographic proof of how an external trainer used its data.

The defaults above are **configurable engineering starting points**, not experimentally calibrated universal thresholds or a statistical significance claim:

| Role | Definition and default |
| --- | --- |
| Primary ranking | Pairwise order agreement of actual per-prompt mean response length with the actual EWS risk score, `median + risk_weight * (p90 - median)`, averaged per prompt. Require >=60%; predicted ties score 0.5, actual pairs differing by <=10% of the larger mean are excluded. This is **not Spearman correlation**. |
| Evidence floor | At least 32 distinct exact prompt identities and 100 comparable pairs. Repeated epochs/siblings do not create new independent identities; pair counts are descriptive, not independent statistical samples. Above 20,000 possible pairs, use a deterministic sample of at most 20,000 before excluding near-ties, bounding CPU overhead without changing global RNG state. Small datasets may explicitly lower the evidence floor, with correspondingly weaker evidence. |
| Gross length/scale protection | Prompt-balanced normalized MAE = `sum(abs(predicted_mean - observed_mean)) / sum(observed_mean)` <=1.0; aggregate predicted/observed mean ratio within [0.25, 4]. This is deliberately a gross-error veto, not a 35% point-accuracy target. |
| Tail protection | Prompt-balanced p90 coverage lower bound >=70% and mean predicted p90 / observed mean <=4. Nominal p90 coverage is 90%; 70% is only a severe-undercoverage guard, not a target. High coverage alone cannot qualify an inflated forecast. |

Any prompt identity with a censored response is excluded from exact mean/rank/MAE calculations. Coverage uses lower/upper bounds: a censored length below predicted p90 has unknown coverage, not a success. Insufficient exact data or bounds straddling the protection threshold cannot certify initial admission. Use held-out workload evidence to choose thresholds before deployment, not to retrofit a favorable result on a completed experiment. The exploratory post-hoc pilot is not a production D2 threshold calibration.

Active monitoring reuses the pre-rollout forecasts (no extra per-epoch fit or post-hoc actor forward), pairs them with completed responses **before DAPO reward filtering**, and pools each five-pass window by prompt identity. One sufficiently supported gross scale/tail violation triggers FCFS at the boundary; ranking failure requires two consecutive supported bad windows by default. Missing/insufficient monitoring evidence also suspends scheduling at the boundary; invalid numeric rows suspend it immediately. No fallback rewrites the completed batch. A suspended checkpoint cannot automatically reactivate: externally fit and validate a new checkpoint, publish a new report, and wait for the next boundary. There is no D1/D2 backend switch.

Monitoring state, incomplete-window statistics, boundary counters, the selected checkpoint path/hash, and the in-progress RL epoch are saved with the trainer. Resume validates counters, grouped statistics, readiness consistency, temporal boundaries, the same quality configuration, and the weight hash **before** loading a candidate onto workers. Missing legacy quality state returns to FCFS admission rather than trusting an old activation flag; a backend mismatch is rejected. Checkpoint snapshots do not alias a live monitor window. Do not overwrite admitted weights in place; saving a changed admitted file also fails explicitly.

The current runtime monitor identifies prompts by unpadded text token IDs. Nonempty `multi_modal_data` suspends quality-mode D2 and uses FCFS before predictor forward, rather than merging different images under the same text identity. Multimodal quality admission is not claimed. A cross-rank candidate-load RPC failure aborts training with readiness cleared; it does not silently continue with mixed heads.

For an already validated matching candidate, `activation.min_epoch=0` and `min_samples=0` explicitly permit admission at startup; the report's epochs then describe the completed historical fitting/validation run. This historical exception is only available at the explicit zero-warm-up startup boundary; later candidate checks must use recent completed epochs of the current run. For short training runs, **20 total RL epochs is a recommendation**, not a hard disable gate: the recipe only warns below 20. Decide whether warm-up and forward overhead can be amortized. D1 fitting and calibration are unchanged.

For exact legacy epoch/sample-only D2 behavior, explicitly set `d2_lifecycle.enable=false` and provide `d2_checkpoint`; no new quality protection is then claimed. Existing D2 launch configurations must either provide a report or select this compatibility mode. D1's `predictor_reorder.epochs` remains the number of ListMLE fitting iterations per update; it is unrelated to these RL epoch boundaries.
