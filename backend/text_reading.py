"""
Text-reading backend: use Bedrock VLM to read digits from images and
write a `text_reading` field into each image's JSON cache.

This module is backend-only (no Streamlit).
"""
import io
import os
import re
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

# Legal digit ranges per device/scale category.
# Keys are matched against scale_class or material_class:
#   - "6_": any scale_class starting with "6_"
#   - "9_": any scale_class starting with "9_"
#   - "radiometer": any material_class containing "radiometer"
#   - "Watermeter": any material_class containing "Watermeter"
# Values are (min_inclusive, max_inclusive) tuples.
LEGAL_DIGIT_RANGE: dict[str, tuple[float, float]] = {
    "6_": (700, 2000),
    "9_": (700, 2000),
    "radiometer": (0.1, 0.25),
    "Watermeter": (0.1, 12),
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
    Returns (clean_digit_or_hscode, flagged_bool).

    - For meter readings ("digit" case): returns only the numeric part
      (digits, optionally with a decimal point), with no words.
    - For HSCODE cases: returns a normalized string like "HSCODE 4707.x0"
      when we can confidently detect it (e.g. "HSCODE 4707.90").
    """
    if not raw_text:
        return "", False

    # Take the last non-empty line as the structured output line.
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return "", False
    last = lines[-1]

    # Determine flagged status from the last line, independent of exact format.
    lowered = last.lower()
    flagged = "flagged" in lowered

    # Prefer the part before " - " as the value area, but fall back to full line.
    value_part = last
    if " - " in last:
        value_part, _ = last.split(" - ", 1)
        value_part = value_part.strip()

    # First, try to detect an explicit HSCODE pattern like "HSCODE 4707.x0".
    # Accept small variations in spacing/case.
    hs_match = re.search(
        r"(HSCODE\s*\d{4}\.\d0)", value_part, flags=re.IGNORECASE
    )
    if hs_match:
        # Normalize spacing and casing: "HSCODE 4707.x0"
        raw_hs = hs_match.group(1).strip()
        parts = raw_hs.split()
        if len(parts) == 2:
            return f"HSCODE {parts[1]}", flagged
        return raw_hs.upper(), flagged

    # If "HSCODE" appears but the strict pattern did not match (e.g. slightly
    # different formatting), try to reconstruct "HSCODE <number>" from the text.
    if "hscode" in value_part.lower():
        # Extract the first numeric-like token after HSCODE.
        after = value_part.split("HSCODE", 1)[1]
        num_match = re.search(r"(\d{3,5}(?:\.\d+)?)", after)
        if num_match:
            return f"HSCODE {num_match.group(1)}", flagged
        # Fall back to returning just "HSCODE" if we cannot find a number.
        return "HSCODE", flagged

    # At this point we assume it's a plain digit reading (no HSCODE).
    # Strip any stray words and keep only the first numeric token (with optional decimal).
    digit_match = re.search(r"(\d{1,4}(?:\.\d+)?)", value_part)
    digit = digit_match.group(1) if digit_match else ""

    return digit, flagged


def _parse_hscode_to_material(hscode_text: str) -> tuple[str | None, str | None]:
    """
    Normalize flexible HSCODE text to a canonical code and material type.

    Rules:
    - Ignore casing, spaces, and punctuation (accept \"HS CODE\", \"HSCODE\", \"470710\",
      \"4707. 10\", etc.).
    - Only accept codes that normalize to 4707.10, 4707.20, or 4707.30.
      * 4707.10 -> OCC
      * 4707.20 -> WHITE
      * 4707.30 -> MIX
    - Reject everything else (return (None, None)).
    """
    if not hscode_text:
        return None, None

    # Remove any leading HSCODE / HS CODE text for robustness.
    lowered = hscode_text.lower()
    lowered = re.sub(r"hs\s*code", "", lowered)
    lowered = re.sub(r"hscode", "", lowered)

    # Keep only digits to ignore spaces and punctuation.
    digits = "".join(ch for ch in lowered if ch.isdigit())
    if len(digits) < 6:
        return None, None

    # Require 4707 as the prefix.
    if not digits.startswith("4707"):
        return None, None

    # Take last two digits as the code suffix.
    suffix = digits[-2:]
    code_map = {
        "10": "OCC",
        "20": "WHITE",
        "30": "MIX",
    }
    material_type = code_map.get(suffix)
    if not material_type:
        return None, None

    normalized_code = f"4707.{suffix}"
    return normalized_code, material_type


def correct_materials_with_hscode(output_path: str) -> None:
    """
    Use HSCODE readings in the text_reading cache to correct material_class.

    For each cached image:
      - If text_reading.digit encodes a valid HSCODE (4707.10/20/30 in flexible formats),
        map to OCC/WHITE/MIX.
      - If current material_class has a different type (OCC/MIX/WHITE), rewrite it to use
        the HSCODE-derived type but keep the same position (inventory/closeup/scale/...).
      - If the code is invalid or non-4707.xx, do nothing.
    """
    json_dir = os.path.join(output_path, "json")
    if not os.path.isdir(json_dir):
        return

    for base in _iter_json_basenames(output_path):
        cached = common.get_cached_labels(output_path, base)
        text_reading = cached.get("text_reading") or {}
        digit = text_reading.get("digit") or ""
        if not digit:
            continue

        # Only consider entries that look like they might contain an HSCODE.
        if "hscode" not in digit.lower() and not any(ch.isdigit() for ch in digit):
            continue

        normalized_code, desired_type = _parse_hscode_to_material(digit)
        if not normalized_code or not desired_type:
            # Reject non-4707.xx codes or unsupported suffixes.
            continue

        material_class = cached.get("material_class")
        if not isinstance(material_class, str) or not material_class:
            continue

        # Expect material_class like "OCC - inventory"
        m = re.match(r"^(\w+)\s*-\s*(.+)$", material_class)
        if not m:
            continue

        current_type, position = m.group(1), m.group(2)
        if current_type not in {"OCC", "MIX", "WHITE"}:
            # Do not override non-paper or unknown types.
            continue

        if current_type == desired_type:
            # Already consistent with HSCODE.
            continue

        new_material_class = f"{desired_type} - {position}"
        common.save_cached_labels(output_path, base, material_class=new_material_class)


def _range_key_for_cached(cached: dict) -> str | None:
    """
    Determine which legal range key applies to this cached record, based on
    its scale_class and material_class.
    """
    material_class = cached.get("material_class") or ""
    scale_class = cached.get("scale_class") or ""

    # Device-based ranges from material_class
    lower_mat = material_class.lower()
    if "radiometer" in lower_mat:
        return "radiometer"
    if "watermeter" in lower_mat:
        return "Watermeter"

    # Scale-based ranges from scale_class (e.g. "6_IT_0", "9_WA_0")
    if isinstance(scale_class, str):
        if scale_class.startswith("6_"):
            return "6_"
        if scale_class.startswith("9_"):
            return "9_"

    return None


def move_out_of_range_and_flagged_to_reject(
    input_folder: str,
    output_path: str,
    legal_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Move images whose digit reading is out of range OR flagged == True to a
    reject folder under output_path.

    - legal_range: mapping like
        {"6_": (min, max), "9_": (min, max), "radiometer": (min, max), "Watermeter": (min, max)}
      If None, uses the module-level LEGAL_DIGIT_RANGE.
    - HSCODE readings are ignored for range checks (but still moved if flagged).
    """
    legal_range = legal_range or LEGAL_DIGIT_RANGE

    json_dir = os.path.join(output_path, "json")
    if not os.path.isdir(json_dir):
        return

    reject_dir = os.path.join(output_path, "reject")
    os.makedirs(reject_dir, exist_ok=True)

    for base in _iter_json_basenames(output_path):
        cached = common.get_cached_labels(output_path, base)
        text_reading = cached.get("text_reading") or {}
        digit_str = text_reading.get("digit") or ""
        flagged = bool(text_reading.get("flagged"))

        # If nothing to evaluate and not flagged, skip.
        if not digit_str and not flagged:
            continue

        # Determine if this record should be rejected.
        reject = False

        # Flagged readings always go to reject.
        # turn this off for now
        # if flagged:
        #     reject = True

        # Range check only applies to numeric digit readings, not HSCODE.
        if not digit_str.upper().startswith("HSCODE"):
            range_key = _range_key_for_cached(cached)
            if range_key and range_key in legal_range:
                # Extract first numeric token from the digit string.
                num_match = re.search(r"(\d+(?:\.\d+)?)", str(digit_str))
                if num_match:
                    try:
                        value = float(num_match.group(1))
                        min_v, max_v = legal_range[range_key]
                        if not (min_v <= value <= max_v):
                            reject = True
                    except ValueError:
                        # Unparsable numeric -> treat as reject for safety.
                        reject = True

        if not reject:
            continue

        img_path = _resolve_image_path(input_folder, base)
        if not img_path:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            image_bytes = common.compress_image(img)
            with open(os.path.join(reject_dir, f"{base}.jpg"), "wb") as f:
                f.write(image_bytes)
            os.remove(img_path)
        except Exception as e:
            print(f"Failed to copy {base} to reject folder: {e}")


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
