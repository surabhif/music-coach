---
title: VocalMatch
emoji: 🎤
colorFrom: rose
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# VocalMatch

Score how closely your singing matches an expert recording of the same song.

VocalMatch isolates vocals (Demucs `htdemucs` on CPU), tracks pitch (librosa pYIN), aligns with DTW, and returns pitch / timing / voice metrics plus note-by-note feedback. It includes an in-browser practice studio, section scoring, Indian classical / raga mode, and SQLite progress history.

> **GitHub Pages cannot host this app** — it needs a Python process, ffmpeg, and several GB of RAM for Demucs. Deploy with **Docker** (Hugging Face Spaces or Render) as described below.

## Screenshots

| Home | Practice studio | Results |
| --- | --- | --- |
| ![Home](screens/home.png) | ![Record](screens/record.png) | ![Results](screens/results.png) |

| Note feedback | Indian / raga | Progress |
| --- | --- | --- |
| ![Notes](screens/notes.png) | ![Indian](screens/indian.png) | ![Progress](screens/progress.png) |

More: `screens/section.png`, `screens/results_key.png`, `screens/results_mobile.png`, `screens/loading.png`, `screens/results_full.png`.

## Features

- **Compare** — upload or record against a library track (`experts/`)
- **Key-tolerant scoring** — global transposition search (default on)
- **Note-by-note feedback** — good / sharp / flat / early / late / missed
- **Practice studio** (`/studio`) — waveform, sections, loop, accompaniment, count-in recording
- **Indian classical mode** — Sa detect, sargam labels, ragas (Yaman, Bhairav, Bhupali, Desh, Kafi, Bilawal), meend/gamak leniency
- **Voice metrics** — steadiness, vibrato, phrasing, dynamics
- **Progress** (`/history`) — SQLite attempt history per singer / song

## Requirements

- Python **3.11+**
- **ffmpeg** on `PATH` (browser recordings → WAV)
- **librosa**, **demucs**, **CPU PyTorch**, Flask, gunicorn (see `requirements.txt`)
- **RAM**: plan on **≥4 GB** available for Demucs (8 GB preferred). First separation downloads the `htdemucs` weights (~80 MB) and caches stems under `cache/`.

Typical processing time on a 4-core CPU for a short demo clip (~10–15 s of audio): **30–90 s** cold (model load + separation); **5–20 s** after stems are cached. Longer songs scale roughly with duration.

## Local setup

```bash
# System packages (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Optional: regenerate synthetic experts + test users (also overwrites demo experts)
python make_test_audio.py

# Dev server (PORT defaults to 7860)
python app.py
# → http://localhost:7860
```

Production-style local run:

```bash
export PORT=7860
export VOCALMATCH_DATA="$(pwd)/data"   # uploads, results, cache, vocalmatch.db
mkdir -p "$VOCALMATCH_DATA"
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 600 app:app
```

### Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `PORT` | `7860` | HTTP port (gunicorn / Flask) |
| `VOCALMATCH_DATA` | app directory | Writable dir for `uploads/`, `results/`, `cache/`, `vocalmatch.db` |
| `VOCALMATCH_EXPERTS` | `./experts` | Expert library (shipped demo tracks live here) |
| `TORCH_HOME` | torch default | Where PyTorch / Demucs weights are cached |

### Tests

```bash
python make_test_audio.py   # creates test_user_*.wav (gitignored) + experts/*.wav
python test_e2e.py
python test_indian.py
python test_sections.py
```

CI runs the same suite via `.github/workflows/test.yml`.

### Demo experts

Shipped under `experts/`:

- `demo_expert.wav` — short western-style demo mix
- `yaman_expert.wav` — Raga Yaman demo

Large `test_user_*.wav` files are **not** committed; regenerate with `make_test_audio.py`.

## Deploy

### Option A — Hugging Face Spaces (Docker) — recommended free/cheap host

HF Spaces free **CPU** machines have enough RAM for Demucs on short clips. GitHub Pages cannot run this stack; Spaces can.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - **SDK**: Docker  
   - **Hardware**: CPU basic (upgrade if you OOM on longer songs)  
   - Visibility: public or private
2. Connect this GitHub repo (Space settings → **Repository** / sync from GitHub), **or** push the repo contents to the Space git remote.
3. Ensure the Space README keeps the YAML frontmatter at the top of this file (`sdk: docker`, `app_port: 7860`). The included `Dockerfile` installs ffmpeg, pins CPU torch, and starts gunicorn on `$PORT`.
4. Wait for the build. First boot downloads Demucs weights into the container (or `/data` if you persist it).
5. Open the Space URL and try **Compare** with `demo_expert.wav`.

**Memory / time on Spaces**

| Resource | Recommendation |
| --- | --- |
| RAM | Free CPU (~16 GB host share) OK for demos; upgrade if OOM |
| CPU | 2+ vCPU; demucs is the bottleneck |
| Disk | Persist `/data` if the platform allows (uploads, cache, DB) |
| Timeout | Gunicorn `--timeout 600`; expect 30–90 s first compare |

Manual steps the owner must do: create the Space (or Render service), link the GitHub repo, and (optional) attach persistent storage for `/data` so caches and history survive rebuilds.

### Option B — Render (Docker Blueprint)

`render.yaml` defines a Docker web service with a persistent disk at `/data`.

1. Go to [render.com](https://render.com) → **New** → **Blueprint** → connect `surabhif/music-coach`.
2. Apply the Blueprint. Choose a plan with **≥2 GB RAM** (prefer **4–8 GB**). Free/starter (~512 MB) will OOM on Demucs.
3. Deploy. Health check: `GET /healthz`.
4. Open the `.onrender.com` URL and run a compare.

Render injects `PORT`; the Dockerfile reads it. `VOCALMATCH_DATA=/data` keeps uploads, results, cache, and SQLite on the disk.

### Option C — Any Docker host

```bash
docker build -t vocalmatch .
docker run --rm -p 7860:7860 -v vocalmatch-data:/data -e PORT=7860 vocalmatch
```

## Project layout

```
app.py              Flask routes (compare, studio, history, healthz)
analysis.py         Demucs → pYIN → DTW → scores / notes / plot
db.py               SQLite progress
paths.py            BASE / DATA / experts / cache paths
voice_synth.py      Synthetic voice used by make_test_audio.py
make_test_audio.py  Regenerates experts + test_user_*.wav
experts/            Demo expert tracks
templates/          Jinja UI
screens/            UI screenshots for docs
Dockerfile          Production image (ffmpeg + gunicorn)
render.yaml         Render Blueprint
requirements.txt    Pinned deps (CPU torch via PyTorch wheel index)
```

## Scoring (summary)

Pipeline: demucs vocal isolation (cached by file hash) → librosa pYIN → DTW on octave-invariant features → scores.

Weights (v3): pitch 35%, timing 15%, steadiness 10%, phrasing 10%, dynamics 10%, coverage 10%, vibrato 5%, micro-stability 5%. Key-tolerant mode searches pitch-class shifts and keeps the global integer shift with lowest median cents error.
