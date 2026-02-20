"""
Shared backend helpers: image compression, Rekognition detect_custom_labels,
and unified per-image label cache. No Streamlit.
"""
import io
import json
import os
from typing import Any

from PIL import Image


def compress_image(img, max_size: int = 1024, quality: int = 85) -> bytes:
    """Compress the image to be under max_size (max dimension in pixels). Returns JPEG bytes."""
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def show_custom_labels(client, model_arn: str, image_bytes: bytes, min_confidence: float = 10):
    """Call Rekognition detect_custom_labels; return list of CustomLabels."""
    response = client.detect_custom_labels(
        Image={"Bytes": image_bytes},
        MinConfidence=min_confidence,
        ProjectVersionArn=model_arn,
    )
    return response["CustomLabels"]


def _json_path(output_path: str, image_name_without_suffix: str) -> str:
    json_dir = os.path.join(output_path, "json")
    return os.path.join(json_dir, f"{image_name_without_suffix}.json")


def get_cached_labels(output_path: str, image_name_without_suffix: str) -> dict[str, Any]:
    """
    Read unified cache file.

    Returns dict with keys:
      - image_name: str
      - scale_labels: list
      - scale_class: str | None  (e.g. "6_IT_0" from scale classification)
      - material_labels: list
      - material_class: str | None  (e.g. "OCC - scale" from material classification)
      - text_reading: dict | None

    Missing keys default to empty list / None; missing file returns
    empty scale_labels/material_labels and no text_reading/scale_class/material_class.
    """
    path = _json_path(output_path, image_name_without_suffix)
    if not os.path.exists(path):
        return {
            "image_name": image_name_without_suffix,
            "scale_labels": [],
            "scale_class": None,
            "material_labels": [],
            "material_class": None,
            "text_reading": None,
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "image_name": data.get("image_name", image_name_without_suffix),
        "scale_labels": data.get("scale_labels", []),
        "scale_class": data.get("scale_class"),
        "material_labels": data.get("material_labels", []),
        "material_class": data.get("material_class"),
        "text_reading": data.get("text_reading"),
    }


def save_cached_labels(
    output_path: str,
    image_name_without_suffix: str,
    scale_labels: list | None = None,
    scale_class: str | None = None,
    material_labels: list | None = None,
    material_class: str | None = None,
    text_reading: dict | None = None,
) -> None:
    """
    Update or create unified cache file. Pass only keys to update; others are preserved.
    """
    path = _json_path(output_path, image_name_without_suffix)
    data = get_cached_labels(output_path, image_name_without_suffix)
    if scale_labels is not None:
        data["scale_labels"] = scale_labels
    if scale_class is not None:
        data["scale_class"] = scale_class
    if material_labels is not None:
        data["material_labels"] = material_labels
    if material_class is not None:
        data["material_class"] = material_class
    if text_reading is not None:
        data["text_reading"] = text_reading
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _resolve_image_path(input_folder: str, base: str) -> str | None:
    """Return path to image in input_folder with common extensions, or None."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(input_folder, base + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def copy_images_to_classified_folders(
    output_path: str,
    input_folder: str,
    material_class_to_dir: dict[str, str],
    scale_class_to_dir: dict[str, str] | None = None,
    progress_callback=None,
) -> None:
    """
    Copy each classified image into its target folder under output_path.
    Reads scale_class and material_class from the json cache (both scales and
    materials write only to cache; actual copying runs here, once at pipeline end).
    Prefer material_class if set, else use scale_class. Run after scales, materials, text_reading.
    """
    scale_class_to_dir = scale_class_to_dir or {}
    json_dir = os.path.join(output_path, "json")
    if not os.path.isdir(json_dir):
        return
    basenames = [
        os.path.splitext(n)[0]
        for n in os.listdir(json_dir)
        if n.lower().endswith(".json")
    ]
    total = len(basenames)
    for current, base in enumerate(basenames, start=1):
        cached = get_cached_labels(output_path, base)
        material_class = cached.get("material_class")
        scale_class = cached.get("scale_class")
        if material_class:
            save_subdir = material_class_to_dir.get(material_class, r"classified/unknown")
        elif scale_class and scale_class in scale_class_to_dir:
            save_subdir = scale_class_to_dir[scale_class]
        else:
            if progress_callback:
                progress_callback(current, total, f"Copy: skipping {base} (no copy destination)")
            continue
        full_save_path = os.path.normpath(os.path.join(output_path, save_subdir))
        img_path = _resolve_image_path(input_folder, base)
        if not img_path:
            if progress_callback:
                progress_callback(current, total, f"Copy: image not found for {base}")
            continue
        os.makedirs(full_save_path, exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        image_bytes = compress_image(img)
        with open(os.path.join(full_save_path, f"{base}.jpg"), "wb") as f:
            f.write(image_bytes)
        if progress_callback:
            progress_callback(current, total, f"Copying {current}/{total} to classified folders...")
