# QC User App 2

Same behavior as QCUserApp, with a clear separation of:

- **Environment / config**: `.env`, `config.py`
- **Backend**: `backend/` (Rekognition, scales, materials; no Streamlit)
- **Display**: `display/streamlit_ui.py` (all Streamlit UI)

## Run

1. Copy `.env.example` to `.env` and adjust if needed.
2. `aws sso login --profile QC-team-app` (or the profile in `.env`).
3. `streamlit run main.py`

## Layout

- `config.py` — loads settings from `.env`
- `main.py` — wires config, backend, and display
- `backend/aws.py` — Rekognition client, check_status, start/stop_model
- `backend/common.py` — compress_image, show_custom_labels, unified label cache (one JSON per image: `image_name`, `scale_labels`, `material_labels`)
- `backend/scales.py` — scale classification (uses cache before calling API)
- `backend/materials.py` — material classification (uses cache before calling API)
- `display/streamlit_ui.py` — folder selection, model status, button, progress, messages

## Configure tips (`.env`)
- Project 1, model 1 refers to the scale model
- Project 2, model 2 refers to the material model
- comment out `streamlit_ui.py` line `82-98` if do not want to run text read

## Function Guide

Below is a summary of where to find key functions and processing steps:

- **aws.py**
  - Rekognition client creation, model management, and status checks.

- **common.py**
  - Image compression, Rekognition API calls, unified per-image label cache (save/read).
  - Copy classified images to target folders.

- **scales.py**
  - `classify_scales`: Handles scale/device classification. Checks cache before Rekognition call.
  - Supporting functions for image orientation correction and post-processing.
  - Scale class assignment logic.

- **materials.py**
  - `classify_materials`: Handles material classification (uses cache before API).
  - Post-processing and assignment of material classes based on predictions.
  - Writes material classification output to cache.

- **display/streamlit_ui.py**
  - Implements the overall application flow.
  - User folder selection, triggers model classification, tracks progress with UI updates.
  - Handles all end-to-end user interaction and coordinates calls to backend modules.

- **config.py**
  - Loads settings from `.env` (API keys, model ARNs, profiles, etc).

For detailed logic and step-by-step flow of the entire pipeline, see `display/streamlit_ui.py`. All backend processing functions (classification, caching, post-processing) are located in `backend/` modules, as noted above.