# Phlebotomy Tube Inversion Tracker

Counts blood-collection-tube inversions (and their speed) from video: a pose model detects the tube's **cap** and **base** keypoints, the cap→base vector becomes an orientation signal, and a hysteresis state machine counts inversions.

## Status

| Stage | Entry point | Output | State |
|---|---|---|---|
| 01 Index + frozen split | `scripts/01_build_index.py` | `splits/video_index.csv` | done |
| Crop box (one-time tool) | `tools/select_crop.py` | `configs/crop.yaml` | done |
| 02 Frame extraction | `scripts/02_extract_frames.py` | `data/frames/round1/` (460 JPGs + `manifest.csv`) | done |
| Keypoint labelling | CVAT (see [Annotation guide](#annotation-guide)) | `data/labels/round1/` | in progress |
| 03 Build YOLO dataset | `scripts/03_build_yolo_dataset.py` | `data/yolo_dataset/` | todo |
| 04 Train pose model | `scripts/04_train.py` | `outputs/runs/` | todo |
| 05 Inference → keypoints | `scripts/05_infer.py` | `data/keypoints/` | todo |
| 06 Tune counter (dev) | `scripts/06_tune_counter.py` | `configs/counter.yaml` | todo |
| 07 Evaluate (test, once) | `scripts/07_evaluate.py` | `outputs/reports/` | todo |
| 08 Demo video | `scripts/08_render_demo.py` | `outputs/demo/` | todo |

**Dataset split** (per video, stratified by folder and count bin, seed 42): 50 `kp_train` / 10 `kp_val` videos are keypoint-labelled for training the pose model; 40 `dev` videos tune the counter; 200 `test` videos are used once for the final numbers.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 5060, CUDA 12.8).

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # expect: True <your GPU>
```

- Python 3.14 (`.python-version`).
- `torch`/`torchvision` are pinned to the PyTorch **cu128** index in `pyproject.toml` (RTX 50-series needs CUDA ≥ 12.8). Keep that pin when adding packages that depend on torch.
- Set the external video location in `configs/paths.yaml` (videos are not stored in the repo). Relative paths in that file resolve from the repo root.

## Running the pipeline

```powershell
uv run scripts/01_build_index.py      # refuses to overwrite the frozen split unless --force
uv run tools/select_crop.py           # draw boxes around where the tube reaches -> configs/crop.yaml
uv run scripts/02_extract_frames.py   # motion-weighted crops of kp_train/kp_val videos
```

- Frames are addressed by their index in a **sequential** decode (`video.iter_frames`) and timed with real timestamps (`t_ms`) — the videos are variable-frame-rate, so never seek by frame number for anything tied to labels.
- Keypoint labels are in **crop coordinates**. Changing `configs/crop.yaml` after labelling invalidates the labels.

## Annotation guide

Self-contained instructions for whoever labels the frames. You only need the frames zip (the JPGs from `data/frames/round1/`) and CVAT — not this repo.

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
2. Send back the zip (name it `round1_labels.zip`). It goes in `data/labels/round1/`.
3. Keep your own copy until it has been received.

Expected contents: `data.yaml` with `kpt_shape: [2, 3]`, and one `.txt` per labelled image with a line `0 cx cy w h  x_cap y_cap v  x_base y_base v` (normalized 0–1; `v` = 2 visible, 1 occluded, 0 outside).

## Layout

```
configs/    paths.yaml, crop.yaml (+ dataset/train/counter settings to come)
splits/     video_index.csv — the frozen video split (committed)
scripts/    numbered pipeline entry points, run in order
tools/      one-off interactive tools (select_crop.py)
src/inversion_tracker/
  config.py     paths from configs/, resolved against the repo root
  video.py      Crop, sequential frame reader with timestamps
  data/         index.py (index + split), frames.py (frame selection/extraction)
tests/      unit tests
data/       generated frames, labels, keypoints (gitignored)
outputs/    training runs, reports, demo videos (gitignored)
```
