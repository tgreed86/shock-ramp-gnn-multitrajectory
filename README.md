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
- Uses GraphSAGE FeatureNet, MeshGraphNet, or conservative FluxGraphNet, with the same
  DEC/PARC inputs, rate loss, and Euler update.

The default is one window per optimizer update, four prediction steps per window,
and a sum of step losses for backward. Reported losses average over steps.

## GNN Architecture

Both native and cache-only training support these independent model choices:

- `model.name="FeatureNet"`: existing GraphSAGE, unchanged. `model.layers=3`
  means **two message-passing blocks plus the MLP head**. `config.json` retains
  this default, and existing FeatureNet checkpoints remain resumable.
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

### FluxGraphNet and open boundaries

Select the architecture in your existing config using `model.name`:

```json
"name": "FluxGraphNet"
```

The same config supports `FeatureNet`, `MeshGraphNet`, and `FluxGraphNet` by
changing only this value. There is no separate config per architecture.
`config_cache_only.json` and `config.json` select cache-only and native data
respectively; both support all three models. Ramp boundary options apply only
to FluxGraphNet and are ignored by the other models. The cache-only config
includes the default ramp-conservation settings for editing.

For example, after changing `model.name` in `config_cache_only.json`:

```bash
python train.py --config config_cache_only.json --out-dir runs/cache-fluxgraphnet-01 --device cuda --epochs 500
```

MeshGraphNet and FluxGraphNet share `message_passing_steps`, `mlp_layers`, and
`hidden`. Start a fresh run when changing architecture; their weights are not
interchangeable. The existing rollout CLI loads FluxGraphNet checkpoints
through the same model factory.

FluxGraphNet shares MeshGraphNet's node encoder, geometric edge encoder, and
residual processor. Its decoder uses both endpoint latents and the processed
edge latent to predict four interface fluxes for `[rho, mx, my, E]`. The two
directions are antisymmetrized into one signed face flux, which is multiplied
by the shared face length, subtracted from the sender, added to the receiver,
and divided by each cell's own area. This conserves area-weighted state on
unequal AMR cells, including coarse–fine interfaces, up to floating-point
roundoff. Supervision remains the existing normalized node-rate loss; no
ground-truth interface flux files are required.

The decoder emits dimensionless flux density `f`. With coarse length `L`,
channel normalization `sigma`, and reference timestep `dt_ref`, its physical
flux density is `f * sigma * L / dt_ref`. If `T = f * face_length / L`, the
normalized cell rate is the signed sum of `T` divided by `cell_area / L²`.
Euler integration still uses `dt / dt_ref`. Normalization means are never
added to fluxes. `predict_rate_and_flux` exposes the raw rate and integrated
interior/exterior transports for diagnostics, before limiting.

As in `shock_ramp/predict_rate/Physics-Aware-cartesian`, separate learned heads
permit boundary exchange while the interior remains conservative. This version
uses actual exposed face lengths instead of a fixed-width band of nodes:

| Model setting | Default | Behavior |
| --- | --- | --- |
| `use_open_boundary_source` | `true` | Enable learned open-boundary fluxes. |
| `open_boundary_sides` | `["left", "right", "top"]` | Mesh-spec sides treated as open. |
| `open_boundary_source_channels` | `[0, 1, 2, 3]` | Channels allowed to enter or leave through open faces. |
| `use_ramp_boundary_source` | `true` | Enable the separate head on remaining wall faces. |
| `ramp_conserved_quantities` | `["density", "energy"]` | Quantities with exactly zero wall flux; all other channels permit learned exchange. |

Each boundary head receives the owner node's processed latent and the exposed
Cartesian face normal. Outward flux is positive; negative predictions permit
inflow. Corner cells can have both open and wall faces. Disable both heads for
a closed-domain conservation experiment.

Choose any subset of `density`, `x_momentum`, `y_momentum`, and `energy` in
`model.ramp_conserved_quantities`. Here conserving density means conserving
area-integrated mass against wall exchange. For example:

```json
"use_ramp_boundary_source": true,
"ramp_conserved_quantities": ["density", "energy"]
```

- `[]`: no quantity is constrained to be conserved at the wall; all four
  wall-flux channels are learned.
- `["density"]`: zero wall mass flux; allow momentum and energy exchange.
- `["density", "energy"]`: the default stationary-wall behavior.
- `["density", "x_momentum", "y_momentum", "energy"]`: zero wall flux in
  every channel.

Conservation here restricts exchange through ramp/wall faces. Interior transport
and the open-boundary configuration are unaffected, so a quantity conserved at
the ramp can still enter or leave at an open boundary. As with the wall head,
this setting applies to any other sides configured as closed. If
`use_ramp_boundary_source=false`, all wall fluxes are zero regardless of this list.

The older `ramp_boundary_source_channels` setting remains supported for existing
configs/checkpoints and lists the opposite: channel indices **allowed** wall
exchange. Use either that setting or `ramp_conserved_quantities`, never both.
Omitting both preserves conservation of mass and energy. Resume retains the
existing requirement that model configuration match the saved checkpoint.

Face coverage is computed from all neighbors, so a coarse–fine interface is
never classified as exterior merely because its node degree differs. Exposed
face midpoints are assigned to the nearest labeled `left/right/bottom/top`
polyline in the trajectory's mesh spec, after the same coordinate rescaling as
the wedge polygon. This is essential for the **curved mapped boundaries** of
the supplied specs. Minimal polygon-only specs infer side labels from outward
segment normals. Geometry is cached per runtime mesh and rebuilt after remeshing.
Conservation refers to the runtime's full Cartesian cells and staircase domain,
not the original mapped native cell areas.

FluxGraphNet also preserves conservation in the surrounding time-stepping path:

- The supplied configs use `train.flux_limiter="local"`: each cell budgets
  harmful face contributions against density/energy floors, configured state
  bounds, and normalized rate/delta bounds. Each interior face uses one scale
  for both cells, preserving equal-and-opposite exchange even across AMR
  interfaces. Boundary faces spend only their owner's budget. A constrained
  cell cannot freeze unrelated faces. Initial states must satisfy the bounds;
  per-cell state clipping remains bypassed. These are linear state bounds,
  not an internal-energy/pressure positivity guarantee.
- Face scales are differentiable, and `loss.flux_raw_rate_weight=0.1` adds
  supervision before limiting: `loss = applied_rate_loss + weight * raw_rate_loss`
  (plus any configured physics residual). Both rate losses use the selected
  MSE/Huber and channel weights. Raw supervision trains saturated predictions
  even when their applied update has no magnitude gradient. It prevents the
  limiter from hiding arbitrarily large raw errors. Local limiting requires
  a positive raw weight. Both options affect only FluxGraphNet.
- Epoch history includes `flux.raw_rate_rms`, `applied_rate_rms`,
  `limited_face_fraction`, `raw_rate_loss`, and `applied_rate_loss` when enabled.
  The main loss includes the auxiliary term and is therefore not directly
  comparable with older runs. RMS and fraction statistics are averaged over
  rollout steps; raw/applied losses are recorded before the auxiliary weight.
- Old checkpoints without these options retain `flux_limiter="global"` and
  zero raw supervision, reproducing their original behavior. Resume rejects
  changing the limiter or loss. Start a fresh run with the supplied options
  to use the correction; the original checkpoint files need no conversion.
- Remeshing uses exact overlap areas on the dyadic lattice: constant
  prolongation on refinement and area-weighted averaging on coarsening. It is
  differentiable and requires the same domain, as enforced by the runtime's
  forced finest refinement at the wedge boundary. It uses a finest-lattice
  ownership map per mesh, so memory grows with `H*W*refine_ratio**(2*max_level)`.
- PARC inputs and residual loss remain available. `loss.blend_weight` must be
  zero: the existing independently sanitized physics baseline is not part of
  the learned flux balance. FeatureNet and MeshGraphNet retain their original
  clipping, blending, and IDW remapping paths.

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

## Optional rate-scaled loss

To balance supervision by each feature's typical rate of change, set this in the
existing config's `loss` section (for any `model.name`):

```json
"rate_loss_scaling": "training_rms",
"rate_loss_scaling_epsilon": 1e-8
```

The default `"none"` preserves the existing objective, including for configs and
checkpoints that omit these options. `"training_rms"` requires `use_huber=false`.
Input/state normalization, predicted rates, and FluxGraphNet conservation remain
unchanged; only the supervised MSE is weighted. Any optional physics residual
penalty retains its existing weighting.

Before the first epoch, the trainer computes the mean squared normalized GT rate
`q_j = mean(((u_next_j - u_current_j) / sigma_j / (dt / dt_ref))**2)` for each
feature using **training-window transitions only**. Each unique transition gets
a fresh runtime mesh selected from its ground-truth starting state; both endpoints
are sampled onto that mesh with the normal GT lookup/interpolation path. The
estimator averages cells within each transition, then averages transitions with
their multiplicity in overlapping training windows. It uses no model predictions
or validation/test transitions. This is a fixed GT reference distribution, not
the evolving distribution of meshes and states in autoregressive training.

Weights are `w_j = (1 / (q_j + epsilon)) / mean_k(1 / (q_k + epsilon))`, and the
supervised loss is `mean_j(w_j * MSE_j)`. The epsilon stabilizes nearly stationary
features. The weights average one, but weighted and unweighted loss values still
represent different objectives and should not be directly compared as performance
scores. Compare per-feature physical errors and rollout quality instead.

The startup pass visits all training transitions even with `--max-batches`, and
adds a one-time preprocessing cost. Weights and mean-square statistics are saved
under `rate_loss_scaling` in each checkpoint and reused unchanged for validation,
test, and resume. Resume rejects missing/inconsistent statistics for enabled runs;
changing the scaling mode requires a new run. Inference without a target does not
need loss weights. Custom callers of `engine.run_epoch` or `engine.model_step`
that compute scaled losses must pass the checkpoint weights on the model device
as `rate_loss_weights`.

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

## Plot Training History

From the project directory, pass a run directory or its `history.json`:

```bash
python eval/plot_training_history.py runs/experiment-01
python eval/plot_training_history.py runs/experiment-01/history.json --log-y --output runs/experiment-01/loss_history.pdf
```

The default output is `loss_history.png` inside the run directory. The plot
shows training and validation loss versus epoch, using the recorded mean loss
per rollout step. Validation points appear only at evaluated epochs; runs
without validation show only training loss. `--log-y` requires positive losses.
Use `--title "My experiment"` to override the title. The script needs only
Matplotlib and Python's standard library, and works without a display on a cluster.

## Rollout Comparison GIFs

Run a trained checkpoint on one chosen cache family and its corresponding original
AMR trajectory. This accepts both flat and nested cache layouts, both GNNs, and
both training data modes. The explicitly selected trajectory may have been
excluded from training.

```bash
python eval/plot_rollout.py \
  --checkpoint runs/experiment-01/best_model.pt \
  --cache-dir /path/to/cache/64x64_Lmax-3/p-5p5_angle-20 \
  --native-pt /path/to/bulkmach_2_output/p_5.5/angle_20/p-5p5_angle-20.pt \
  --native-map-dir /path/to/bulkmach_2_output/p_5.5/angle_20/plot \
  --device cuda
```

`--native-map-dir` can be omitted when the map files are under `plot/` beside the
`.pt` file. **Exact native cell geometry requires the corresponding
`DMR.plot.*.2d.map.hdf5` files.** The `.pt` stores centers but not cell corners;
the script errors if the required maps are unavailable. Original field values
always come from `.pt`; map files supply only geometry. Map time, cell indices,
and mapped centers are checked against the `.pt`. Native and model refinement
levels use separate colorbars because their refinement ratios differ.

By default, `rollout_plots/` is created beside the checkpoint being tested.
For the example above, results go to
`runs/experiment-01/rollout_plots/p-5p5_angle-20/`:

- `density.gif`, `x_momentum.gif`, `y_momentum.gif`, `energy.gif`: each has
  three synchronized panels—model prediction, uniform L3 GT, and original AMR GT.
- `mesh.gif`: predicted adaptive mesh and exact original AMR mesh.
- `rollout.json`: input paths, checkpoint/model/data mode, normalization, physical
  times, frame indices, color limits, and geometry provenance.
- `rollout_data.h5`: numerical fields, exact mesh geometry, common-grid samples,
  and masks for computing RMSE, MAE, SSIM, or other metrics offline at every step.

The uniform reference uses the checkpoint's maximum cache level (L3 for the
supplied configs). All panels use model coordinates, including the native map
corners transformed using the `.pt`'s recorded physical/normalized bounds.
Each field shares one fixed color scale across all three panels and all frames.
GIF playback uses constant FPS while titles show the actual variable physical
times. Rendering follows the quadrilateral, fixed-scale, and streamed-GIF
approach of the original `utils/make_shock_ramp_gifs.py`; that external script
is not a runtime dependency.

Initialization and evolution match the checkpoint's training/evaluation path:

- **Cache-only checkpoint:** uniform L3 GT drives the first mesh prediction.
  Initial state on that mesh is looked up from the level-specific caches, with
  the existing finest-cache IDW fallback for missing cells.
- **Native checkpoint:** native GT drives the first mesh prediction; initial
  features follow that checkpoint's cache/IDW settings.
- Subsequent states are fully autoregressive. The script reuses `model_step`,
  physical time intervals, checkpoint `dt_ref`/normalization, remeshing cadence,
  prediction remapping, and state/rate guardrails. It uses evaluation mode
  (dropout disabled). Future GT is read only for comparison panels.
- This is **one continuous rollout**, without resets at training-window
  boundaries. The initial frame is labeled as initialized GT. A prediction at
  `t+1` is shown on the mesh selected from state `t`, as in training.

No training configuration file or native dataset root is needed. The `.pt`
provides physical timestamps, so the timestamp-export JSON is unnecessary for
this evaluation script. Checkpoint normalization is never recomputed. Cache
provenance and mesh specs are validated; cache-only trajectories that were
present in training are also checked against their saved content identities.
Mesh specs default to the saved directory when available, otherwise the bundled
`mesh_specs/`; use `--mesh-specs /path/to/mesh_specs` to override.

For a shorter rollout, append `--start 0 --steps 10`. The default runs to the
trajectory's last snapshot. `--frame-stride 2` keeps every second display frame
plus the final frame, while still executing **every** intermediate model step.
Other display options include `--fps 10`, `--dpi 150`, `--cmap viridis`, and
`--xlim MIN MAX --ylim MIN MAX` for zooming. Mesh lines below two screen pixels
are hidden by level to avoid aliasing; `--min-edge-pixels 0` draws every edge.
Use `--output-dir rollout_plots/another-run` for separate checkpoints or ranges;
existing outputs require `--overwrite`.

Install the updated `requirements.txt` for ImageIO/Pillow GIF support. No display
or ffmpeg is required. Numerical data is streamed into compressed HDF5, then GIFs
are rendered from that export. Outputs are staged before replacing completed
results. GPU inference is supported, while plotting/export runs on CPU.

### Numerical Rollout Export

`plot_rollout.py` always saves `rollout_data.h5` alongside the GIFs. This export
includes **every simulated step**, independent of `--frame-stride`, plot zoom,
DPI, colormap, or GIF frame rate. Step zero is the initialized state, not a model
prediction; steps 1 through N are the autoregressive predictions. Corresponding
native map files are therefore required for every exported step, including steps
omitted from the GIF. No model rerun or access to the original data is needed to
compute metrics from the completed export.

The schema (`format="shock-ramp-rollout-v1"`, `complete=true`) contains:

| HDF5 path | Contents |
| --- | --- |
| `rollout_step`, `snapshot_index` | Relative rollout step and original trajectory index |
| `time`, `elapsed_time`, `dt`, `is_initial` | Physical timestamps, time since initialization, step intervals, initial-state flag |
| `feature_names` | Channel order: density, x momentum, y momentum, energy |
| `normalization/{mu,sigma,dt_ref}` | Training normalization and timestep scale from the checkpoint |
| `grid/{x,y,bbox}` | Fixed finest-grid center coordinates and domain bounds |
| `grid/{prediction,uniform_gt,native_gt}` | Float32 arrays shaped `(N+1, Hfine, Wfine, 4)`, in original field units |
| `grid/{prediction,uniform_gt,native_gt}_valid` | Boolean geometry masks shaped `(N+1, Hfine, Wfine)` |
| `reference_ranges/{uniform_gt,native_gt}/{min,max,data_range}` | Per-feature GT ranges over all exported grid samples and steps |
| `uniform_mesh/{centers,levels}` | Static finest-cache geometry |
| `snapshots/IIIIII/prediction/{centers,levels,values}` | Actual adaptive prediction mesh and carried field values |
| `snapshots/IIIIII/uniform_gt/values` | Original finest-cache values in `uniform_mesh` order |
| `snapshots/IIIIII/native_gt/{centers,levels,values,vertices}` | Native fields and exact mapped quadrilaterals; native bounds/indices are also saved |
| `config_json`, `metadata_json` | Resolved configuration and run/input provenance |

`IIIIII` is the zero-padded original snapshot index. Predictions are saved after
the same physical-state guardrails used to carry state into the next training
step. Fields are stored numerically with lossless compression, not reconstructed
from GIF colors. Grid row zero is at the bottom of the domain (increasing y).
For the supplied 64×64/L3 configuration, the comparison grid is 512×512.

The comparison grid takes the value of the cell containing each grid center:
adaptive model cells for predictions, cached cells for uniform GT, and exact
mapped native cells for original GT. There is no smoothing, IDW, or extrapolation
in this export sampling. This point-sampled comparison is distinct from training
loss on the changing mesh, or averaging errors equally over native AMR cells.
Raw cell values and vertices are retained for alternative sampling/area weighting.

For RMSE/MAE, use the intersection of prediction and reference masks; invalid
pixels are NaN. Report valid-region coverage when comparing steps, because that
region can change. Example for per-feature error against uniform GT at step 1:

```python
import h5py
import numpy as np

with h5py.File("rollout_data.h5", "r") as f:
    step = 1
    prediction = f["grid/prediction"][step].astype(np.float64)
    reference = f["grid/uniform_gt"][step].astype(np.float64)
    valid = f["grid/prediction_valid"][step] & f["grid/uniform_gt_valid"][step]
    error = prediction[valid] - reference[valid]
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
```

Replace `uniform_gt` with `native_gt` for the original-simulation comparison,
and loop over `rollout_step` to build histories. Handle an empty intersection
explicitly. For SSIM, preserve the 2-D grid, use a fixed per-feature `data_range`,
and average only over windows entirely inside the common valid region. Filling
masked regions with zeros and including those windows would distort the score.
The saved GT ranges are available for choosing that range; a constant feature
has zero range and needs an explicit metric convention. These files supply the
data for metrics; metric calculation/plotting is not performed by the GIF script.

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
project; this project consumes the already generated caches. Continuous rollout
visualization is provided separately by `eval/plot_rollout.py`.

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
