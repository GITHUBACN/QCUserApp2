"""
Text-reading backend: use Bedrock VLM to read digits from images and
write a `text_reading` field into each image's JSON cache.

This module is backend-only (no Streamlit).
"""
import io
import os
from typing import Iterable, Tuple

import boto3
from PIL import Image

from backend import common

# Thresholds for target labels (from materials.py cf_thrsd_dict)
TEXT_READING_THRESHOLDS = {
    "sign": 55,
    "oldWatermeter": 50,
    "newWatermeter": 50,
    "radiometer": 50,
    # LCD_SCREEN variants use 60 (from _cropped_image_from_rekognition default)
    "LCD_SCREEN": 60.0,
}


def _get_bedrock_client(region: str, profile_name: str | None = None):
    """
    Create a bedrock-runtime client.

    Uses the same profile mechanism as the rest of the app when provided,
    otherwise falls back to the default credential chain.
    If AWS_BEARER_TOKEN_BEDROCK is set in the environment, the SDK uses it
    for Bedrock auth (same as the playground notebook), bypassing IAM.
    """
    # Normalize bearer token: strip quotes/whitespace so it starts with ABSK (required prefix)
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip().strip('"').strip("'")
    if token:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client(service_name="bedrock-runtime", region_name=region)
    return boto3.client(service_name="bedrock-runtime", region_name=region)


def _cropped_image_from_rekognition(
    image_path: str,
    labels: list,
    min_confidence: float = 60.0,
    min_width: int = 60,
    min_height: int = 30,
) -> Image.Image:
    """
    Crop the LCD screen area from the image using Rekognition labels.
    Falls back to the full image if no suitable label is found.
    """
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    target_label_names = ["LCD_SCREEN_0", "LCD_SCREEN_0_MAIN"]
    mx_confidence = min_confidence
    mx_label = None

    for label in labels or []:
        if (
            label.get("Name") in target_label_names
            and label.get("Confidence", 0) > mx_confidence
            and "Geometry" in label
            and "BoundingBox" in label["Geometry"]
        ):
            mx_label = label
            mx_confidence = label["Confidence"]

    if not mx_label:
        return image

    box = mx_label["Geometry"]["BoundingBox"]

    left = int(box["Left"] * img_w)
    top = int(box["Top"] * img_h)
    right = int((box["Left"] + box["Width"]) * img_w)
    bottom = int((box["Top"] + box["Height"]) * img_h)

    if right - left < min_width:
        extra = min_width - (right - left)
        left = max(0, left - extra // 2)
        right = min(img_w, right + extra // 2)

    if bottom - top < min_height:
        extra = min_height - (bottom - top)
        top = max(0, top - extra // 2)
        bottom = min(img_h, bottom + extra // 2)

    return image.crop((left, top, right, bottom))


def _construct_conversation(prompt: str, image_bytes: bytes) -> list[dict]:
    """
    Build a Bedrock Converse conversation payload with one user message
    containing text + image.
    """
    img_type = "jpeg"
    return [
        {
            "role": "user",
            "content": [
                {"text": prompt},
                {
                    "image": {
                        "format": img_type,
                        "source": {"bytes": image_bytes},
                    }
                },
            ],
        }
    ]


def _invoke_bedrock(
    client,
    model_id: str,
    conversation: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Call Bedrock converse API and return the text output.
    """
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    for part in content:
        if "text" in part:
            return part["text"]
    return ""


def _parse_text_reading_output(raw_text: str) -> Tuple[str, bool]:
    """
    Parse the model output.

    Expectation: a final line like "{digit/HSCODE} - {flagged/None}".
    Returns (digit_without_HSCODE, flagged_bool).
    """
    if not raw_text:
        return "", False

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    if " - " not in last:
        return "", False

    left, right = last.split(" - ", 1)
    digit = left.strip()
    if digit.upper().startswith("HSCODE"):
        parts = digit.split(maxsplit=1)
        digit = parts[1] if len(parts) == 2 else ""

    flagged_str = right.strip().lower()
    flagged = "flagged" in flagged_str
    return digit, flagged


def _iter_json_basenames(output_path: str) -> Iterable[str]:
    """
    Yield basenames (without .json) for all json files under output_path/json.
    """
    json_dir = os.path.join(output_path, "json")
    if not os.path.isdir(json_dir):
        return []
    for name in os.listdir(json_dir):
        if name.lower().endswith(".json"):
            yield os.path.splitext(name)[0]


def _resolve_image_path(input_folder: str, image_name_without_suffix: str) -> str | None:
    """
    Try to find the corresponding image file in input_folder with common extensions.
    """
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for ext in exts:
        candidate = os.path.join(input_folder, image_name_without_suffix + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _has_target_label(cached: dict) -> bool:
    """
    Check if the image has any of the target labels that require text reading
    AND the label confidence is above the threshold:
    - "sign" >= 55
    - "oldWatermeter" or "newWatermeter" >= 50
    - "radiometer" >= 50
    - "LCD_SCREEN" variants >= 60
    """
    # Check scale_labels
    for label in cached.get("scale_labels", []):
        label_name = label.get("Name", "")
        confidence = label.get("Confidence", 0)
        
        # Check LCD_SCREEN variants
        if label_name.startswith("LCD_SCREEN") and confidence >= 60.0:
            return True
        
        # Check other target labels
        threshold = TEXT_READING_THRESHOLDS.get(label_name)
        if threshold is not None and confidence >= threshold:
            return True
    
    # Check material_labels
    for label in cached.get("material_labels", []):
        label_name = label.get("Name", "")
        confidence = label.get("Confidence", 0)
        
        threshold = TEXT_READING_THRESHOLDS.get(label_name)
        if threshold is not None and confidence >= threshold:
            return True
    
    return False


def add_text_reading_to_jsons(
    input_folder: str,
    output_path: str,
    text_config,
    aws_profile: str | None = None,
    progress_callback=None,
) -> None:
    """
    For each cached image JSON in output_path/json, read the image, optionally crop
    to the LCD screen, call Bedrock VLM, and write a `text_reading` field into
    the JSON:

        \"text_reading\": {\"digit\": <str>, \"flagged\": <bool>}

    This does not modify any Streamlit UI.
    """
    client = _get_bedrock_client(region=text_config.region, profile_name=aws_profile)

    basenames = list(_iter_json_basenames(output_path))
    total = len(basenames)
    current = 0

    for base in basenames:
        cached = common.get_cached_labels(output_path, base)

        # Skip if we already have text_reading
        if cached.get("text_reading") is not None:
            current += 1
            if progress_callback:
                progress_callback(current, total, f"Skipping {base} (already has text_reading)")
            continue

        # Only process images with target labels: sign, Watermeter, radiometer, LCD_SCREEN
        if not _has_target_label(cached):
            current += 1
            if progress_callback:
                progress_callback(current, total, f"Skipping {base} (no target labels)")
            continue

        img_path = _resolve_image_path(input_folder, base)
        if not img_path:
            current += 1
            if progress_callback:
                progress_callback(current, total, f"Image not found for {base}, skipping")
            continue

        scale_labels = cached.get("scale_labels", [])
        try:
            cropped = _cropped_image_from_rekognition(img_path, scale_labels)
            img_bytes = common.compress_image(cropped)
            conversation = _construct_conversation(text_config.prompt, img_bytes)
            raw = _invoke_bedrock(client, text_config.model_id, conversation)
            digit, flagged = _parse_text_reading_output(raw)
            text_reading = {"digit": digit, "flagged": bool(flagged)}
            common.save_cached_labels(output_path, base, text_reading=text_reading)
        except Exception as e:
            # Log to stdout; callers can inspect JSONs later.
            print(f"text_reading failed for {base}: {e}")

        current += 1
        if progress_callback:
            progress_callback(current, total, f"Processed {current}/{total} images for text_reading")

