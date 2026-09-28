# Phlebotomy Tube Inversion Tracker

Counts blood-collection-tube inversions (and their speed) from video: a pose model detects the tube's **cap** and **base** keypoints, the cap→base vector becomes an orientation signal, and a hysteresis state machine counts inversions.

## Status

| Stage | Entry point | Output | State |
|---|---|---|---|
| 01 Index + frozen split | `scripts/01_build_index.py` | `splits/video_index.csv` | done |
| Crop box (one-time tool) | `tools/select_crop.py` | `configs/crop.yaml` | done |
| 02 Frame extraction | `scripts/02_extract_frames.py` | `data/images/train/` (460 JPGs + `manifest.csv`) | done |
| Keypoint labelling | CVAT (see [Annotation guide](#annotation-guide)) | `data/round1_labels/` | done |
| 03 Build YOLO dataset | `scripts/03_build_yolo_dataset.py` | `data/yolo_dataset/` (383 train / 77 val) | done |
| 04 Train pose model | `scripts/04_train.py` + `configs/train.yaml` | `outputs/runs/pose/r1_s_640/` | done (YOLO26s-pose, 640; keypoint-level eval skipped) |
| Preview on a video | `tools/preview_video.py` | `outputs/previews/` | done |
| 05 Inference → keypoints | `scripts/05_infer.py` | `data/keypoints/r1_s_640/` (dev + test) | done |
| 06a Signal plots | `scripts/06a_plot_signals.py` | `outputs/reports/signals/` | done |
| 06 Tune counter (dev) | `scripts/06_tune_counter.py` + `configs/counter_grid.yaml` | `configs/counter.yaml` (frozen) | done |
| 07 Evaluate (test) | `scripts/07_evaluate.py` | `outputs/reports/eval/r1_s_640/test/` | done |
| 08 Demo video | `scripts/08_render_demo.py` | `outputs/demo/` | todo |

**Dataset split** (per video, stratified by folder and count bin, seed 42): 50 `kp_train` / 10 `kp_val` videos are keypoint-labelled for training the pose model; 40 `dev` videos tune the counter; 200 `test` videos are held out for the final numbers.

## Results

The frozen counter was scored against the file-name counts. The unrounded count is flips / 2, so "within 0.5" accepts e.g. 1.5 for a true count of 2.

| Split | Videos | Exact | Within 0.5 | Within 1 | MAE | Bias |
|---|---|---|---|---|---|---|
| Dev (tuning) | 40 | 67.5% | 95.0% | 100% | 0.33 | −0.33 |
| **Test** | **200** | **60.0%** | **90.0%** | **98.5%** | **0.41** | **−0.30** |

- **Test errors:**
  - 68 videos undercounted by 1 and 9 overcounted by 1;
  - 2 undercounted by 2 and 1 overcounted by 2.
- **Likely cause of the undercount:** the first inversion after pickup is often made towards the camera and barely registers in 2D.
- **More detail:** per-folder and per-count tables, the confusion matrix and signal plots of every miss are in `outputs/reports/eval/r1_s_640/test/`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 5060, CUDA 12.8).

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # expect: True <your GPU>
```

- Python 3.14 (`.python-version`).
- `torch`/`torchvision` are pinned to the PyTorch **cu128** index in `pyproject.toml` (RTX 50-series needs CUDA ≥ 12.8). Keep that pin when adding packages that depend on torch.
- Set the external video location in `configs/paths.yaml` (videos are not stored in the repo). Relative paths in that file resolve from the repo root.

Note: Before running the pipeline, please check the paths.yaml config files in the /config directory. You may have to update the video_root value to the folder where your actual dataset exists. The video filenames must be of the format <video_id>_<inversion_count> 
## Running the pipeline

```powershell
uv run scripts/01_build_index.py      # refuses to overwrite the frozen split unless --force
uv run tools/select_crop.py           # draw boxes around where the tube reaches -> configs/crop.yaml
uv run scripts/02_extract_frames.py   # motion-weighted crops of kp_train/kp_val videos
# label in CVAT, export to data/round1_labels/
uv run scripts/03_build_yolo_dataset.py   # padded boxes, train/val by video -> data/yolo_dataset/
uv run scripts/04_train.py                # settings in configs/train.yaml; override with --model/--name/--epochs/--batch/--imgsz
uv run tools/preview_video.py 64          # annotated preview of one video (default: random kp_val)
uv run scripts/05_infer.py                # keypoints for every frame of dev + test -> data/keypoints/<run>/
uv run scripts/06a_plot_signals.py        # s(t) plots for 12 dev videos (--all for every one)
uv run scripts/06_tune_counter.py --write # grid search on dev; freezes the best settings to configs/counter.yaml
uv run scripts/07_evaluate.py --splits dev   # sanity check: reproduces dev_score
uv run scripts/07_evaluate.py             # test; refuses to overwrite an existing evaluation without --force
```

- Frames are addressed by their index in a **sequential** decode (`video.iter_frames`) and timed with real timestamps (`t_ms`) — the videos are variable-frame-rate, so never seek by frame number for anything tied to labels.
- Keypoint labels are in **crop coordinates**. Changing `configs/crop.yaml` after labelling invalidates the labels.
- Stage 03 rebuilds CVAT's boxes (tight box around the keypoints + 45 px padding, ±75 px around a single point) and writes outside (`v=0`) points as `0 0 0`. Overlays for a visual check go to `outputs/reports/label_check/`.
- Stage 04 fine-tunes `yolo26s-pose.pt` at `imgsz=640` with 180° rotation and up/down + left/right flips (`flip_idx [0, 1]`: flips never swap cap and base). Anything that runs the model must apply the same crop first.
- Ultralytics YOLO is **AGPL-3.0**, and so are weights fine-tuned from it.

### How counting works (stages 05–07)

1. **Keypoints (stage 05):** every frame is cropped and run through the model. Frames with no detection stay as NaN rows, so the time axis has no holes.
2. **Signal (`counting/signal.py`):**
   - **Clean:** frames where a keypoint is below the confidence cut-off are dropped, and gaps of 0.3 s or less are interpolated.
   - **Vector:** `v = cap − base` for every frame.
   - **Main swing direction:** `u` is the first principal component of `v` (PCA).
   - **Signal:** `s = (v·u − centre) / L_ref`. The centre is the middle of the swing range, `L_ref` is the tube's full visible length, and + is the side the tube starts on.
   - **Smooth:** Savitzky-Golay filter.
3. **Counter (`counting/state_machine.py`):**
   - A Schmitt trigger at ±T: each crossing to the opposite side is a flip.
   - Optionally, an isolated first or last flip (pickup / put-down) is dropped.
   - The count is flips / 2, rounded down, so only completed over-and-back inversions count.
4. **Tuning (stage 06):** grid search over `configs/counter_grid.yaml` on dev. The winner is chosen by exact accuracy, then by its grid neighbours' average. It's frozen in `configs/counter.yaml`, and stage 07 reads only that file.

## Annotation guide

Self-contained instructions for whoever labels the frames. You only need the frames zip (the JPGs from `data/images/train/`) and CVAT — not this repo.

### 1. Install and start CVAT (one time)

*Windows:* CVAT runs in Docker via WSL 2.

1. `wsl --install --no-distribution` in an admin PowerShell, then reboot. Optional resource limits in `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=6GB
   processors=4

   [experimental]
   sparseVhd=true
   autoMemoryReclaim=gradual
   ```
2. Install Docker Desktop (WSL 2 backend). Disable "start on sign-in".
3. Clone CVAT at the pinned version and start it:
   ```powershell
   git clone --config core.autocrlf=input https://github.com/cvat-ai/cvat C:\tools\cvat
   cd C:\tools\cvat; git checkout v2.76.0
   docker compose up -d
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   ```
4. Open http://localhost:8080 in Chrome and log in.

Day to day: start Docker Desktop → `docker compose up -d` in `C:\tools\cvat`; stop with `docker compose stop`.
**Never run `docker compose down -v` before exporting labels** — it deletes all CVAT data.

### 2. Create the project and skeleton

1. **Projects → + → Create a new project**, name `tube-inversion`.
2. **Setup skeleton**, label name **`tube`**:
   - **Add point** → right-click it → **Configure** → name **`cap`**. Create `cap` **first** (point order = exported keypoint order).
   - Add a second point below it, named **`base`**.
   - **Add edge** between them. Keep the template vertical (cap on top).
3. **Continue → Submit & Open**.

### 3. Create the task

1. In the project: **+ → Create a new task**, name `round1` (labels are inherited from the project).
2. **Select files → My computer** → add **all the JPGs** (not `manifest.csv`).
3. **Advanced configuration**: **Image quality = 100**; sorting **lexicographical**; leave segment size empty (one job).
4. **Submit & Open**, then open the job.

### 4. Label every frame

- Left toolbar: **Draw new skeleton** → label `tube` → **Shape** (not Track). Drag a box around the tube; the template appears inside it. Drag **`cap`** onto the centre of the cap's top face and **`base`** onto the centre of the tube's rounded bottom.
- `N` repeats the last drawing mode · `F` / `D` next / previous frame · scroll to zoom · **`Ctrl+S` save often**. `F1` lists all shortcuts.
- One skeleton per frame (one tube per image).

**Labelling rules — consistency matters more than pixel precision:**

| Situation | What to do |
|---|---|
| Point clearly visible | Place it. Leave as visible. |
| **Motion blur** | Place each point at the **middle of its smear** (not the leading/trailing edge). Blurred is still **visible** — do not mark it occluded. |
| Point hidden but you can infer where it is (behind a hand, base inside the container) | Place it at your best guess, then hover the point and press **`Q`** (occluded). |
| Point outside the image | Hover the point and press **`O`** (outside). |
| **Can't tell which end is the cap**, both ends blurred into one streak, or tube mostly hidden *and* heavily blurred | **Skip the frame** (leave it unlabelled). Never guess the cap/base direction — a swapped cap/base is the one error that breaks counting. |

Label most blurred frames — the model needs them. If you are skipping more than ~10–15% of frames, you are being too strict.

### 5. Export and hand back

1. **Menu → Export task dataset** → format **Ultralytics YOLO Pose 1.0**, **Save images: off**.
2. Send back the zip (name it `round1_labels.zip`). It goes in `data/round1_labels/`.
3. Keep your own copy until it has been received.

Expected contents: `data.yaml` with `kpt_shape: [2, 3]`, and one `.txt` per labelled image with a line `0 cx cy w h  x_cap y_cap v  x_base y_base v` (normalized 0–1; `v` = 2 visible, 1 occluded, 0 outside).

## Layout

```
configs/    paths.yaml, crop.yaml, train.yaml, counter_grid.yaml, counter.yaml (frozen)
splits/     video_index.csv — the frozen video split (committed)
scripts/    numbered pipeline entry points, run in order
tools/      standalone tools (select_crop.py, preview_video.py)
src/inversion_tracker/
  config.py     paths from configs/, resolved against the repo root
  video.py      Crop, sequential frame reader with timestamps
  inference.py  per-frame pose inference on a whole video
  data_preprocessing/
                index.py (index + split), frames.py (frame selection/extraction),
                yolo.py (CVAT export -> YOLO pose dataset)
  counting/     signal.py (keypoints -> s(t)), state_machine.py (s(t) -> count),
                metrics.py (scores), plots.py (signal diagnostic figure)
data/       generated frames, labels, keypoints (gitignored)
outputs/    training runs, reports, demo videos (gitignored)
```
