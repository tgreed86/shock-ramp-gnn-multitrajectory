# Shock-Ramp GNN: Multi-Trajectory Runtime Training

An independent, focused runtime-meshing trainer derived from Physics-Aware.
No imports point into the original project. The default configuration targets
CUDA; CPU and MPS are supported for local testing.

## Cache-Only Training

`config_cache_only.json` selects `data.source="uniform_cache"`. This mode trains
entirely from the interpolated L0-L3 HDF5 state fields. It does **not** discover,
load, or stat the original trajectory `.pt` files. The small bundled `.pt` mesh
specifications are still required; they describe geometry, not trajectories.
`config.json` retains the native initialization/statistics mode described below.

Copy these inputs to the cluster:

1. This project, including `mesh_specs/`, `config_cache_only.json`, and
   `cache_timestamps.json`.
2. The `p-..._angle-.../` cache families beneath a common cache root. HDF5 files
   may be directly inside each trajectory directory or in its `uniform_meshes/`
   subdirectory. Both training modes and the timestamp exporter detect the layout
   automatically, independently for each trajectory. A family with HDF5 caches
   in both locations is rejected to avoid choosing an unintended copy.

Set `train.runtime_mesh.multires_gt_lookup.root` to that root and choose a writable
`train.save_dir` (or pass `--out-dir`). The cache-only config has no `data.root`.
Relative timestamp/spec paths resolve against the config file. Keep all nine
families in the catalog if you want arbitrary exclusions; excluded families are
identified by directory name and are not opened. Unknown exclusions are errors.

For the cluster layout `64x64_Lmax-3_uniform_meshes/p-1p0_angle-20/*.h5`, set
`train.runtime_mesh.multires_gt_lookup.root` to the absolute path of
`64x64_Lmax-3_uniform_meshes`. Do not append a trajectory name or `uniform_meshes`.
No layout setting or file rearrangement is needed.

```bash
python train.py --config config_cache_only.json --dry-run
python train.py --config config_cache_only.json --device cpu --epochs 1 --max-batches 1 --out-dir /tmp/shock-cache-smoke
python train.py --config config_cache_only.json --device cuda --epochs 500 --out-dir runs/cache-only-01
```

The supplied cache-only configuration retains the current `p-5p5_angle-30`
exclusion and five-snapshot windows: 654 train, 77 validation, and 39 test
windows. `--max-batches` is only for smoke tests. The random overlapping-window
split, global stride alignment, source ordering, and exclusion behavior are
unchanged. Source ordering is retained from the exported relative native names;
sorting cache-directory names directly would change split membership.

The HDF5 files do not contain physical timestamps. `cache_timestamps.json` contains
all 903 timestamps exported from the nine original simulations, preserving their
variable intervals and original floating-point values. It contains no state
fields. The export is a **one-time preparation step**, already completed for the
supplied data. Training reads only this JSON and calculates `dt_ref` from included
trajectories with the existing median rule. It never substitutes unit intervals.
For a new dataset, generate another metadata file where its native data exists:

```bash
python export_cache_times.py --config config.json --output new_cache_timestamps.json
```

The exporter reads native files with memory mapping, extracts timestamps and
source provenance, and verifies them against all four caches. It exports all
trajectories independently of training exclusions and refuses to overwrite an
existing metadata file. Set `data.timestamps_path` to the resulting JSON. No
export step runs during training, dry-run, or resume.

### Cache-only numerical semantics

- The first mesh in each window is predicted from the finest uniform cache
  (L3 here), with either `gradient_fast` or `gradient`.
- After selecting a mesh, initial GNN state and per-step targets use the existing
  level-specific L0-L3 lookup. The initial L3 predictor input does not replace
  those level-specific targets.
- Unmatched boundary cells use IDW **from the same-timestep L3 cache**, with the
  configured interpolation backend and `k`. Its static neighbor indices/weights
  are reused for that mesh. `fallback_to_idw=false` instead makes misses errors.
  Native GT is never an interpolation source in this mode.
- Normalization uses L3 cells only, with the existing training-window snapshot
  multiplicities, sample variance correction, and momentum-sigma option. Other
  levels and duplicate HDF5 timestep representations are not counted again.
- Ramp angle comes from the trajectory identity; the distance feature is fitted
  from the static L3 centers using the existing fitting algorithm.
- Subsequent meshes use autoregressive predicted fields; remapping, losses,
  physics operators, Euler updates, and remeshing cadence remain the same.

This is a **different experiment** from native initialization and normalization.
Interpolated L3 fields can change gradients and initial refinement decisions;
uniform-cell statistics change spatial weighting; the ramp fit and boundary
targets also change. Preserving physical timestamps does not make these other
changes numerically equivalent. These choices are printed at startup and recorded
in the split manifest/checkpoint format.

The existing global 2048 MiB CPU / 512 MiB CUDA feature budgets and reusable CUDA
lookup plans also cover finest-cache feature reads. Static geometry and IDW maps
are outside the feature-byte budgets, as are the existing index tables.
Startup validates metadata, geometry, all feature dataset shapes, and the exact
time count. It hashes the HDF5 files in bounded blocks for portable checkpoint
identity; this incurs a sequential read of the cache files on every startup.
Loaded feature arrays are checked for finiteness.

Cache-only checkpoints use `shock-ramp-uniform-cache-v1`. Resume allows relocating
identical cache/spec/timestamp files and changing performance-only options.
Changed cache contents, timestamps, source order, or split membership are rejected.
Native-mode checkpoints cannot be resumed as cache-only experiments.

```bash
python train.py --config config_cache_only.json --resume runs/cache-only-01/last_model.pt --out-dir runs/cache-only-01 --epochs 600
```

## What It Does

The original/native mode (`config.json`, or `data.source="native"`) works as follows:

- Discovers `p_*/angle_*/*.pt` beneath `data.root`.
- Excludes named trajectories **before loading, normalization, or any split**.
- Retains the original seeded random window split, including overlapping windows.
- Selects wedge geometry and geometry input features for each trajectory.
- Selects that trajectory's L0-L3 uniform HDF5 caches using local timestep indices.
- Predicts the first mesh from the native GT state, not a uniform-L3 substitute.
- Maps initial GT onto that mesh, then predicts autoregressively. Subsequent mesh
  decisions use predicted features, not future GT.
- Remeshes every `train.runtime_mesh.update_every_steps` rollout steps, starting
  at step zero; remaps predictions only when the mesh is rebuilt.
- Looks up GT targets at every step; explicitly interpolates only missing cells
  when `fallback_to_idw=true`. Missing/corrupt files are errors, not silent fallback.
- Uses GraphSAGE FeatureNet by default, or optional MeshGraphNet, with the same
  DEC/PARC inputs, rate loss, and Euler update.

The default is one window per optimizer update, four prediction steps per window,
and a sum of step losses for backward. Reported losses average over steps.

## GNN Architecture

Both native and cache-only training support these independent model choices:

- `model.name="FeatureNet"`: existing GraphSAGE, unchanged. `model.layers=3`
  means **two message-passing blocks plus the MLP head**. Both supplied configs
  retain this default, and existing FeatureNet checkpoints remain resumable.
- `model.name="MeshGraphNet"`: node/edge encoders, independent residual graph
  processor blocks, and an MLP rate decoder. Changing only the name works with
  either supplied config; processor depth defaults to 15 and MLP depth to two
  hidden layers. For explicit settings, replace the `model` section with:

```json
"model": {
  "name": "MeshGraphNet",
  "hidden": 128,
  "message_passing_steps": 15,
  "mlp_layers": 2,
  "dropout": 0.0,
  "predict_type": "rate",
  "activation": "relu",
  "layernorm_eps": 1e-6
}
```

`hidden` sets the node/edge latent width. `mlp_layers` counts hidden layers in
each encoder, processor MLP, and decoder, before its output linear layer.
`message_passing_steps` sets the number of separately parameterized processor
blocks. `dropout` applies after each hidden activation (the example disables it;
changing only the name retains your existing dropout). Activation settings and
`layernorm_eps` are shared with FeatureNet. MeshGraphNet always uses residual
node/edge updates and LayerNorm after latent MLPs; the decoder has neither an
output activation nor output LayerNorm. GraphSAGE's `layers`, `use_skip`,
`skip_type`, and `use_layernorm` settings apply only to FeatureNet and are ignored
by MeshGraphNet.

The implementation follows the [MeshGraphNets encode-process-decode architecture](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/core_model.py),
adapted to this project's cell-centered graph. Each directed face edge gets three
geometric inputs: sender-minus-receiver center displacement `(dx, dy)` and its
Euclidean length, all divided by `max(coarse_dx, coarse_dy)`. This fixed scaling
replaces the reference implementation's learned edge normalization. Edge MLPs
use both endpoint latents and edge latents; node MLPs receive the sum of incoming
new edge messages before residual addition. This is an architecture option in
the existing shock-ramp experiment, not a reproduction of the paper's full
training recipe. Node features (including optional PARC terms), targets, time
scaling, remeshing, and the random window split retain their existing definitions.

Start a new run after changing architecture, for example after editing the
cache-only config:

```bash
python train.py --config config_cache_only.json --out-dir runs/cache-meshgraphnet-01 --epochs 500
```

Use `--config config.json` for native data instead. Architecture changes require
new weights; resume rejects a different model configuration. MeshGraphNet keeps
edge latents at every processor step and generally needs more GPU memory than
the default GraphSAGE. Reduce `message_passing_steps` or `hidden` if needed and
smoke-test with `--max-batches 1 --epochs 1` before a full cluster run.

## Trajectory Exclusions

Edit `config.json`, for example:

```json
"excluded_trajectories": [
  "p-5p5_angle-30",
  "p-10_angle-40"
]
```

Use exact filename stems without `.pt`. An empty list includes all nine
trajectories. Unknown names and excluding every trajectory are errors.

Excluded trajectories do not enter training, internal validation, the built-in
test split, normalization, or timestep scaling. Reserve them for separate
post-training rollout evaluation. No trajectories are excluded by default.

The retained split is **not** a held-out-trajectory experiment: all included
simulations may contribute overlapping windows to train/val/test. Fractions
`split.train` and `split.val` determine rounded counts; test receives the remainder,
as in the original trainer (`split.test` is informational). Membership is fixed
by the seed; windows are shuffled within training epochs. Empty validation/test
sets are supported. Global stride alignment and boundary filtering match the
original split implementation.

## Paths and Data Sharing

The project contains only its code, tests, documentation, and three small L3 mesh
specifications. Raw trajectories and 36 HDF5 caches are **not copied**.
The supplied config references their existing absolute locations.

On the cluster, change these three configuration values:

1. `data.root`: directory containing the pressure/angle trajectory folders.
2. `train.runtime_mesh.multires_gt_lookup.root`: directory containing the
   `p-..._angle-.../uniform_meshes/` cache families.
3. `train.save_dir`: a new writable run directory.

`mesh.starting_mesh_path` points to the bundled `mesh_specs/` directory. Relative
paths resolve against the config file, not the shell's current directory.
Environment variables and `~` are expanded in path settings.

Source `.pt` files are loaded read-only using memory mapping. `max_open_trajectories`
limits simultaneously open cache families (four files each, default 9). Feature
cache keys include the trajectory, refinement level, and physical local timestep.
The two HDF5 representations of a timestep share one in-memory entry. Wedge
constraints are retained per geometry. No files in the data/cache roots are written.

## Runtime Acceleration

The supplied config enables these paths automatically on CUDA:

```json
"runtime_mesh": {
  "adjacency_backend": "auto",
  "reuse_unchanged_mesh": true,
  "multires_gt_lookup": {
    "backend": "auto",
    "cpu_cache_mb": 2048,
    "cuda_cache_mb": 512
  }
}
```

These are additions to the existing `train.runtime_mesh` settings, not a complete
replacement config. The `[RUNTIME]` startup line reports the selected paths.

- `cpu_cache_mb` and `cuda_cache_mb` are **global MiB budgets across all trajectories**,
  not per-file/per-trajectory allocations. Features load lazily and least-recently
  used entries are evicted. A zero budget disables retention on that device;
  arrays larger than the budget are used without being retained. These budgets
  exclude model memory, active working tensors, geometry/index tables, and the
  CUDA allocator's reserved memory. They are not total process-memory limits.
- `max_cached_feature_steps` is obsolete. Old configs are accepted with a warning;
  that setting is ignored in favor of the byte budgets.
- Lookup `backend="auto"` uses dense integer cell-to-row tables, cached feature
  tensors, and gathers on CUDA when training on CUDA. CPU and MPS use CPU lookup
  and transfer assembled results as needed. `backend="cpu"` forces the CPU path
  for comparison; `"cuda"` requires a CUDA training device. HDF5 cache misses still
  require disk/CPU reads and, on CUDA, transfer complete level-specific arrays.
- Cell-to-cache lookup plans are reused for the lifetime of an unchanged mesh.
  Missing boundary cells still use the existing IDW fallback, with the same
  `fallback_to_idw` behavior. Model inputs/targets remain timestep-specific.
- Adjacency `"auto"` uses vectorized tensor rasterization on CUDA and the reference
  CPU implementation on CPU/MPS. `"cpu"` forces the reference implementation;
  `"tensor"` explicitly enables the tensor implementation on CPU or CUDA. It
  preserves directed edge order, coarse-fine interfaces, and wedge holes.
- `reuse_unchanged_mesh=true` skips materialization, connectivity construction,
  and state remapping when the new constrained refinement masks equal the prior
  masks within that window. The predictor still runs on schedule. `rebuilds` in
  logs continues to count scheduled mesh updates, even when geometry is reused.

These options do not alter the split, model, losses, physics operators, or remeshing
cadence. Cache budgets/backends and mesh reuse can change when resuming an existing
new-project checkpoint; numerical runtime settings must still match. For a CUDA
A/B comparison, force both backends to `cpu` and disable mesh reuse, then compare
warmed-up runs with the same seed, split, model, and update interval. That compares
CPU vs CUDA paths, not the old entry-limited feature cache. Keep
`debug.sync_runtime_timing=false` when measuring end-to-end throughput.

## Run

Use the existing training Python environment, or install `requirements.txt` with
a PyTorch build appropriate for your machine. Only load trusted project `.pt`
files: these contain Python-serialized data.

```bash
python train.py --config config.json --dry-run
python train.py --config config.json --out-dir runs/experiment-01
```

The default run uses CUDA for 500 epochs. `--dry-run` validates every included
source, cache family, mesh spec, and split without training or writing outputs.

Local smoke test:

```bash
python train.py --device cpu --epochs 1 --max-batches 2 --out-dir /tmp/shock-smoke
```

`--max-batches` caps each split separately and is recorded in the run config.
Remove it for real training. `--device mps` is available on supported Macs.

Resume a **new-project** checkpoint:

```bash
python train.py --resume runs/experiment-01/last_model.pt --out-dir runs/experiment-01
```

`epochs` is the total target epoch, not the additional number. Resume restores
model, optimizer, scheduler, normalization, and CPU/CUDA RNG state, and checks
the trajectory split and numerical configuration. Old Physics-Aware checkpoints
are not accepted by this entry point.

Run outputs include resolved `config.json`, exact `split_manifest.json`,
`history.json` with per-trajectory metrics, `last_model.pt`, `best_model.pt` when
validation runs, and `test_metrics.json`. Last checkpoints are written every
epoch. Final built-in testing uses the last model, matching the original behavior.
An existing nonempty output directory is never overwritten without `--resume`.

## Supported Scope

Supported: `gradient_fast` (default) and `gradient`, wedge lookup geometry,
multiresolution GT lookup or explicit IDW, exact torch/CUDA IDW or CPU KD-tree IDW,
GraphSAGE with optional skips/layer normalization, MeshGraphNet, DEC/PARC, Euler integration,
normalization, gradient clipping, AdamW, optional plateau scheduling, checkpointing.

Not carried over: CNN mesh predictors/training, offline precompute training,
warm-start precomputes, graph chunking, MLS, RK4, adapters, attention/other GNN
families, deprecated dataset wrappers, plotting suites, backups, experiments,
old checkpoints, Git history, or Git remotes. Unsupported legacy modes are
rejected rather than silently emulated. Cache generation remains in the original
project; this project consumes the already generated caches. Held-out full-rollout
visualization is not implemented in this focused training entry point.

## Timing and Verification

Progress bars are enabled by default for each training, validation, and test
pass. They show epoch/split, completed/total batches, elapsed time, estimated
remaining time, processing rate, and running mean loss/MAE. With `--max-batches`,
the total reflects the smoke-test cap. Bars write to stdout, including redirected
cluster logs, and close cleanly on errors or interruptions.

Control them with `debug.progress_bar` (true/false) and
`debug.progress_bar_mininterval` (minimum refresh interval in seconds, default
5.0). Set the latter to 1.0 for more frequent terminal updates, or increase it
for quieter cluster logs. These controls do not enable GPU synchronization.
Existing batch messages are printed without overwriting the bar. Set
`debug.progress_every_batches` to 0 if you want only the bar and epoch summaries.

`debug.progress_every_batches` prints source, local timestep, mesh/remapping/model/
backward times, rebuilds, and cache misses. CUDA/MPS operations are asynchronous;
enable `debug.sync_runtime_timing` for diagnostic per-bucket timing. Keep it false
for production throughput measurements. Model operations, DEC, tensor edge
attributes, exact IDW, adjacency construction, and cached GT gathers use CUDA
when selected. HDF5 reads, first-load feature staging, and portions of runtime
control still use CPU; this is not a fully GPU-resident pipeline.

```bash
python -m unittest discover -s tests -v
```

Tests cover exclusions, legacy split parity, window-weighted normalization,
per-source/local-time lookup, byte-budget LRU eviction, wrong-cache rejection,
canonical-time sharing, lookup-plan reuse, IDW fallback, gradient propagation,
both predictors, mesh reuse, CPU/tensor adjacency parity, and remeshing cadence. CUDA/MPS training
tests run when those devices are available, otherwise they are explicitly skipped.
See `PORTING.md` for implementation differences and verification details.
