# Frozen D2 admission and monitoring

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
