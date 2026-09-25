# Phlebotomy Tube Inversion Tracker

Counts blood-collection-tube inversions (and their speed) from video: a pose model detects the tube's **cap** and **base** keypoints, the cap→base vector becomes an orientation signal, and a hysteresis state machine counts inversions.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 5060, CUDA 12.8).

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # expect: True <your GPU>
```

- Python 3.14 (`.python-version`).
- `torch`/`torchvision` are pinned to the PyTorch **cu128** index in `pyproject.toml` (RTX 50-series needs CUDA ≥ 12.8). Keep that pin when adding packages that depend on torch.
- Set the external video location in `configs/paths.yaml` (videos are not stored in the repo).

## Annotation tool: CVAT (runs outside this repo)

*For Windows Users* CVAT runs in Docker via WSL 2. One-time minimal setup if you don't have WSL or Docker:

1. `wsl --install --no-distribution` (admin), reboot. Optional limits in `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=6GB
   processors=4

   [experimental]
   sparseVhd=true
   autoMemoryReclaim=gradual
   ```
2. Install Docker Desktop (WSL 2 backend). Disable "start on sign-in".

Running CVAT:

1. Clone CVAT at the pinned version and start it:
   ```powershell
   git clone --config core.autocrlf=input https://github.com/cvat-ai/cvat C:\tools\cvat
   cd C:\tools\cvat; git checkout v2.76.0
   docker compose up -d
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   ```
2. Open http://localhost:8080 (Chrome).

Day to day: start Docker Desktop → `docker compose up -d` in `C:\tools\cvat`; stop with `docker compose stop`.
**Never run `docker compose down -v` before exporting labels** — it deletes all CVAT data.

## Layout

```
configs/    paths, crop box, dataset/train/counter settings
splits/     video_index.csv — the frozen video split (committed)
scripts/    numbered pipeline entry points, run in order
src/inversion_tracker/   pipeline code
tests/      unit tests
data/       generated frames, labels, keypoints (gitignored)
outputs/    training runs, reports, demo videos (gitignored)
```
