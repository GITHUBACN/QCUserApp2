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

# Config
in .env
- Project 1, model 1 refers to the scale model
- Project 2, model 2 refers to the material model

- comment out streamlit_ui.py 82-98 if do not want to run text read
