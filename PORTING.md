# Porting Notes

## Separation

The original Physics-Aware directory is a read-only source for this port.
The new project is not a copy of its worktree: selected numerical functions were
extracted by dependency, retaining their algorithms while dropping unrelated
top-level functions, obsolete comments, and entire unused training paths.
`kernel_provenance.json` records source files and retained symbols. It is
provenance, not a runtime dependency.

The new entry point is about 200 lines rather than the original 13,000+ line
trainer. One shared epoch loop serves training/validation/test. Source objects
replace global concatenated-timestep routing, and cache readers are keyed by
source path. Three angle-specific specs are the only binary assets bundled.

## Preserved Numerical Behavior

- Original global-stride window enumeration, cross-source boundary filtering,
  seeded NumPy permutation, and rounded train/validation/remainder counts.
- Native-GT step-zero predictor input with coarse parents inferred from centers.
- Gradient-fast filling, dilation, thresholds, hierarchical gating, and wedge
  constraints; the legacy gradient predictor remains optional.
- GraphSAGE layer/head structure and state-dict parameter names for the retained
  configuration. Removed score heads had no role in the trained field prediction.
- DEC operators, PARC input units, normalized rate targets, Euler updates,
  state/rate guardrails, summed window loss for backward, AdamW updates.
- Window-weighted feature statistics from training windows only. Statistics are
  accumulated per snapshot with multiplicity instead of rereading overlapping
  windows. The accumulation order/precision differs slightly.

## Deliberate Differences

- Optional `model.name="MeshGraphNet"` adds an encode-process-decode model with
  directed geometric edge features, summed incoming messages, independent
  processor blocks, and residual node/edge updates. Its edge geometry uses fixed
  coarse-cell scaling; latent MLPs use LayerNorm. This changes the learned model,
  while retaining the existing node features, rate targets, DEC/PARC, Euler
  integration, and random window split. Native/cache-only data selection is
  independent. FeatureNet remains the default with unchanged parameters and
  forward semantics. A different architecture requires a new run, not resume.
- An optional `data.source="uniform_cache"` mode uses the finest uniform cache for
  step-zero mesh input, train-window-weighted normalization, the fitted ramp
  context, and missing-cell IDW. L0-L3 caches still supply level-specific state
  and targets. These are explicit experiment changes, not native-GT numerical
  parity claims. Native mode remains available and retains its prior behavior.
- Cache-only physical times are exported once to JSON. That exporter is the only
  new entry point that reads native trajectories. The cache-only loader needs no
  native root and validates provenance against the saved metadata and bundled
  mesh specs. Relative source names preserve the original enumeration order.
- Cache-only checkpoints have a separate format and content-based cache/spec/time
  identity. Storage paths can change on resume; experiment/data identity cannot.

- IDW neighbor selection is still nondifferentiable, but application of the fixed
  weights to predicted features now preserves autograd. The original helper had
  a `no_grad` decorator that cut this connection for interpolated cells.
  GT lookup/remapping and discrete mesh prediction still run without gradients.
- Edge attributes use their existing tensor implementation on the selected
  device rather than moving geometry to CPU first. CUDA now uses a vectorized
  adjacency builder with the reference directed edge ordering; the CPU builder
  remains selectable and is the automatic path on CPU/MPS.
- Startup rejects non-increasing time values instead of silently substituting a
  timestep. `dt_ref` is the median of valid within-trajectory time differences;
  artificial cross-trajectory placeholders are not included.
- Raw data stays memory-mapped on CPU. Uniform GT features are cached lazily under
  separate global CPU/CUDA byte budgets. Canonical physical-timestep keys share
  the duplicated HDF5 initial/target representation. HDF5 cache provenance is
  validated before training. CUDA lookup uses device-resident integer tables and
  reuses geometry-only lookup plans; CPU/MPS retain a CPU lookup path.
- Identical constrained refinement masks reuse the prior mesh within a window.
  The predictor still runs on schedule; identical-mesh remapping is an identity
  with gradients preserved. This is not cross-window mesh-prediction memoization.
- GPU computation uses float32 without AMP. The baseline model forward already
  forced float32; exact training trajectories are not promised to be bitwise
  identical (sampler order, statistics accumulation, and remapping gradients differ).
- Final testing remains the original last-model, random-window test, not a new
  held-out-trajectory/best-checkpoint protocol. External held-out testing is separate.
- Checkpoints use a versioned format and save optimizer/scheduler/RNG/split state.
  No old run directories, checkpoints, or remote repository settings were copied.

## Verification Performed

- MeshGraphNet addition: 48 tests passed and 12 accelerator-dependent tests were
  skipped locally. Coverage includes directed edge geometry, node/edge permutation
  equivariance, incoming sum/residual semantics, gradients, empty edges, both
  predictors and data modes, FeatureNet factory/forward compatibility, exact CPU
  resume parity, and rejection of changed architecture/depth on resume.
- A real cache-only CPU MeshGraphNet smoke run (width 16, two processor blocks,
  two hidden layers per MLP) trained, validated, tested, and resumed for a second
  epoch with native trajectory loading/stat calls blocked. Each pass used one
  five-snapshot window; the original 654/77/39 split was retained. This verifies
  integration, not convergence or default-size GPU memory/throughput.
- Cache-only addition: 33 tests passed and eight accelerator-dependent tests were
  skipped locally. Coverage includes absent native directories, physical timestamp
  export/validation, original source ordering and window membership, Lmax
  normalization, cached initialization, boundary IDW/map reuse, both predictors,
  relocated resume, and rejection of changed times/cache contents or data modes.
- The real cache-only dry-run retained 654/77/39 windows with the supplied
  `p-5p5_angle-30` exclusion. A real-data CPU smoke run trained, validated, tested,
  and resumed for another epoch while native trajectory `torch.load` and `stat`
  calls were explicitly blocked. Each pass was capped at one window; this is
  functional verification, not a convergence or throughput benchmark.
- All nine physical timestamp sequences (903 timestamps, 894 intervals) were
  exported to `cache_timestamps.json`. CUDA/MPS validation of the new mode remains
  pending availability of those devices.

- Dry-run checks passed for all nine real simulations and all 36 caches.
- With the supplied five-snapshot windows and stride one: 737 train, 87 validation,
  43 test windows, before exclusions.
- A real-data CPU smoke run completed training, validation, testing, checkpoint
  saving, and resuming for another epoch.
- Six real-data gradient-fast mesh comparisons (three ramp angles, initial and
  subsequent predicted-state step) matched the original mesh/edge outputs exactly.
- Simplified FeatureNet forward outputs matched the original with shared weights,
  including optional skip and layer-normalization configurations.
- Automated synthetic tests exercise both mesh predictors and trajectory routing.
- Runtime acceleration tests cover global byte budgets, disabled/oversized cache
  entries, reader eviction, physical-timestep sharing, source isolation, lookup
  misses/fallback, plan reuse, unchanged-mesh gradients, and ordered adjacency
  parity on mixed-level/holey meshes. CPU training parameter updates match across
  adjacency backends in the synthetic comparison.
- Real-file acceleration checks compared 96 duplicated timestep pairs across the
  eight currently included trajectories. All matched exactly. Lookup values and
  ordered edges also matched the reference on real meshes for all three angles.
- CUDA and MPS were unavailable in this execution environment. Their conditional
  tests are included, but accelerator performance and correctness still need a
  cluster/device smoke run before full-scale training.
