"""
Text-reading backend: use Bedrock VLM to read digits from images and
write a `text_reading` field into each image's JSON cache.

This module is backend-only (no Streamlit).
"""
import io
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import boto3
from PIL import Image, ImageOps

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

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _get_bedrock_client(region: str, profile_name: str | None = None):
    """
    Create a bedrock-runtime client.

    If AWS_BEARER_TOKEN_BEDROCK is set, use a plain boto3 client (same as crop.ipynb).
    Otherwise fall back to an IAM profile when provided.
    """
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return boto3.client(service_name="bedrock-runtime", region_name=region)
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
    Crop the highest-confidence LCD_SCREEN_0 / LCD_SCREEN_0_MAIN bbox.
    Falls back to the full image if no suitable label is found.
    """
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
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


def _image_to_jpeg_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image to JPEG bytes (quality 85, no resize) — same as crop.ipynb."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


_ROTATION_VARIANTS: list[tuple[str, int]] = [
    ("upStraight",       0),
    ("counterClockwise", 90),
    ("upsideDown",      180),
    ("clockwise",       270),
]

_UPRIGHT_PROMPT = (
    "I am showing you 4 versions of the same image, each rotated differently:\n"
    "- Image 1: original orientation\n"
    "- Image 2: rotated 90° counter-clockwise from original\n"
    "- Image 3: rotated 180° from original\n"
    "- Image 4: rotated 90° clockwise from original\n\n"
    "Look at the digits or text visible in each image. "
    "Which single image number (1, 2, 3, or 4) shows the content most upright and readable?\n\n"
    "Reply with ONLY a single digit: 1, 2, 3, or 4."
)


def _find_upright_rotation(
    img: Image.Image,
    client,
    model_id: str,
) -> tuple[Image.Image, str]:
    """
    Send all 4 rotations to Bedrock in one call.
    Returns (upright_image, rotation_label). Falls back to original if response is unexpected.
    """
    rotated_imgs = [img.rotate(deg, expand=True) for _, deg in _ROTATION_VARIANTS]
    rotated_bytes = [_image_to_jpeg_bytes(r) for r in rotated_imgs]

    content: list[dict] = [{"text": _UPRIGHT_PROMPT}]
    for b in rotated_bytes:
        content.append({"image": {"format": "jpeg", "source": {"bytes": b}}})

    conversation = [{"role": "user", "content": content}]
    raw = _invoke_bedrock(client, model_id, conversation, max_tokens=10, temperature=0.0)

    match = re.search(r"[1-4]", raw or "")
    idx = (int(match.group()) - 1) if match else 0
    label, _ = _ROTATION_VARIANTS[idx]
    return rotated_imgs[idx], label


def _construct_conversation(prompt: str, image_bytes: bytes) -> list[dict]:
    """Build a Bedrock Converse payload with one user message containing text + image."""
    return [
        {
            "role": "user",
            "content": [
                {"text": prompt},
                {
                    "image": {
                        "format": "jpeg",
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
    max_tokens: int = 1500,
    temperature: float = 0.0,
) -> str:
    """Call Bedrock Converse API and return the text output."""
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    for part in content:
        if "text" in part:
            return part["text"]
    return ""



def _process_text_reading_worker(
    base: str,
    img_path: str,
    scale_labels: list,
    text_config,
    aws_profile: str | None,
    local_state,
) -> dict:
    """Run one text-reading inference and return a structured result dict."""
    try:
        if not hasattr(local_state, "client"):
            local_state.client = _get_bedrock_client(
                region=text_config.region, profile_name=aws_profile
            )

        cropped = _cropped_image_from_rekognition(img_path, scale_labels)
        upright, rotation = _find_upright_rotation(
            cropped, local_state.client, text_config.model_id
        )
        img_bytes = _image_to_jpeg_bytes(upright)

        conversation = _construct_conversation(text_config.prompt, img_bytes)
        raw = _invoke_bedrock(
            local_state.client,
            text_config.model_id,
            conversation,
            max_tokens=text_config.max_tokens,
            temperature=text_config.temperature,
        )
        digit, _, flagged = _parse_text_reading_output(raw)
        return {"base": base, "digit": digit, "rotation": rotation, "flagged": bool(flagged), "error": None}
    except Exception as e:
        return {"base": base, "digit": "", "rotation": "", "flagged": False, "error": str(e)}


def _parse_text_reading_output(raw_text: str):
    """
    Parse the model output.

    Expected final line: "{digit/HSCODE} - {rotation} - {flagged/None}".
    Returns (clean_digit_or_hscode, rotation, flagged_bool).

    rotation is one of upStraight, counterClockwise, clockwise, upsideDown
    when recognized (spacing/case variants are normalized).
    """
    if not raw_text:
        return "", "", False

    _ROT_MAP = {
        "upstraight": "upStraight",
        "counterclockwise": "counterClockwise",
        "clockwise": "clockwise",
        "upsidedown": "upsideDown",
        "ccw": "counterClockwise",
        "cw": "clockwise",
    }

    _KNOWN_ROTATION = frozenset(
        {"upStraight", "counterClockwise", "clockwise", "upsideDown"}
    )

    def _compact(s: str) -> str:
        return re.sub(r"[\s_-]+", "", s.strip().lower())

    def _canonical_rotation(tok: str) -> str:
        """Only return a known rotation; never echo stray tokens like HSCODE."""
        if not tok or not tok.strip():
            return ""
        c = _compact(tok)
        if c in _ROT_MAP:
            return _ROT_MAP[c]
        t = tok.strip()
        return t if t in _KNOWN_ROTATION else ""

    def _parse_value(value_part: str) -> str:
        if not value_part:
            return ""
        hs_match = re.search(r"(HSCODE\s*\d{4}\.\d0)", value_part, flags=re.IGNORECASE)
        if hs_match:
            raw_hs = hs_match.group(1).strip()
            parts = raw_hs.split()
            if len(parts) == 2:
                return f"HSCODE {parts[1]}"
            return raw_hs.upper()
        if "hscode" in value_part.lower():
            after = value_part.split("HSCODE", 1)[1]
            num_match = re.search(r"(\d{3,5}(?:\.\d+)?)", after)
            if num_match:
                return f"HSCODE {num_match.group(1)}"
            return "HSCODE"
        digit_match = re.search(r"(\d{1,4}(?:\.\d+)?)", value_part)
        return digit_match.group(1) if digit_match else ""

    def _normalize_line(line: str) -> str:
        return line.strip().replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")

    def _rotation_hint(line: str) -> str:
        """Recover rotation from a free-text token when structured parsing fails."""
        t = line.lower()
        if re.search(r"counter\s*-?\s*clockwise|\bccw\b", t):
            return "counterClockwise"
        if re.search(r"\bupside\s*-?\s*down\b", t) or "upsidedown" in _compact(line):
            return "upsideDown"
        if re.search(r"\bup\s*-?\s*straight\b", t) or _compact(line).find("upstraight") >= 0:
            return "upStraight"
        if re.search(r"\bclockwise\b", t) and "counter" not in t:
            return "clockwise"
        return ""

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return "", "", False
    last = _normalize_line(lines[-1])

    parts = [p.strip() for p in re.split(r"\s*-\s*", last) if p.strip()]

    if len(parts) >= 3:
        flag_part = parts[-1]
        rot_raw = parts[-2]
        value_part = " - ".join(parts[:-2])
        flagged = "flagged" in flag_part.lower()
        rot = _canonical_rotation(rot_raw) or _rotation_hint(rot_raw)
        return _parse_value(value_part), rot, flagged

    if len(parts) == 2:
        value_part, second = parts[0], parts[1]
        sk = _compact(second)
        if sk in _ROT_MAP:
            return _parse_value(value_part), _ROT_MAP[sk], False
        sl = second.lower()
        if sl in ("flagged", "none"):
            return _parse_value(value_part), "", sl == "flagged"
        flagged = "flagged" in sl
        rot = _canonical_rotation(second) or _rotation_hint(second)
        return _parse_value(value_part), rot, flagged

    flagged = "flagged" in last.lower()
    return _parse_value(last), _rotation_hint(last), flagged


def _parse_hscode_to_material(hscode_text: str) -> tuple[str | None, str | None]:
    """
    Normalize flexible HSCODE text to a canonical code and material type.
    Only accepts codes normalizing to 4707.10 (OCC), 4707.20 (WHITE), 4707.30 (MIX).
    """
    if not hscode_text:
        return None, None

    match = re.search(r"4707[.\s]?(\d{2})", hscode_text, flags=re.IGNORECASE)
    if not match:
        return None, None

    suffix = match.group(1)
    material_type = {"10": "OCC", "20": "WHITE", "30": "MIX"}.get(suffix)
    if not material_type:
        return None, None

    return f"4707.{suffix}", material_type


def correct_materials_with_hscode(output_path: str) -> None:
    """
    Use HSCODE readings in the text_reading cache to correct material_class.

    For each cached image:
      - If text_reading.digit encodes a valid HSCODE (4707.10/20/30), map to OCC/WHITE/MIX.
      - If current material_class has a different type, rewrite it keeping the same position.
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

        if "hscode" not in digit.lower() and not any(ch.isdigit() for ch in digit):
            continue

        normalized_code, desired_type = _parse_hscode_to_material(digit)
        if not normalized_code or not desired_type:
            continue

        material_class = cached.get("material_class")
        if not isinstance(material_class, str) or not material_class:
            continue

        m = re.match(r"^(\w+)\s*-\s*(.+)$", material_class)
        if not m:
            continue

        current_type, position = m.group(1), m.group(2)
        if current_type not in {"OCC", "MIX", "WHITE"} or current_type == desired_type:
            continue

        common.save_cached_labels(output_path, base, material_class=f"{desired_type} - {position}")


def _range_key_for_cached(cached: dict) -> str | None:
    """Return the LEGAL_DIGIT_RANGE key for this cached record, or None."""
    material_class = cached.get("material_class") or ""
    scale_class = cached.get("scale_class") or ""

    lower_mat = material_class.lower()
    if "radiometer" in lower_mat:
        return "radiometer"
    if "watermeter" in lower_mat:
        return "Watermeter"

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
    Copy out-of-range digit readings to output_path/reject. Input folder is never modified.
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
        rotation = text_reading.get("rotation") or ""

        if not digit_str:
            continue

        reject = False
        if not digit_str.upper().startswith("HSCODE"):
            range_key = _range_key_for_cached(cached)
            if range_key and range_key in legal_range:
                num_match = re.search(r"(\d+(?:\.\d+)?)", str(digit_str))
                if num_match:
                    try:
                        value = float(num_match.group(1))
                        min_v, max_v = legal_range[range_key]
                        if not (min_v <= value <= max_v):
                            reject = True
                    except ValueError:
                        reject = True

        if not reject:
            continue

        img_path = _resolve_image_path(input_folder, base)
        if not img_path:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = common._rotate_by_text_reading(img, rotation)
            img_bytes = _image_to_jpeg_bytes(img)
            with open(os.path.join(reject_dir, f"{base}.jpg"), "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            print(f"Failed to copy {base} to reject folder: {e}")


def _iter_json_basenames(output_path: str) -> Iterable[str]:
    """Yield basenames (without .json) for all json files under output_path/json."""
    json_dir = os.path.join(output_path, "json")
    if not os.path.isdir(json_dir):
        return []
    for name in os.listdir(json_dir):
        if name.lower().endswith(".json"):
            yield os.path.splitext(name)[0]


def _resolve_image_path(input_folder: str, base: str) -> str | None:
    """Return the first matching image path for base in input_folder, or None."""
    for ext in _IMAGE_EXTENSIONS:
        candidate = os.path.join(input_folder, base + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _has_target_label(cached: dict) -> bool:
    """
    Return True if the image has any target label above its confidence threshold:
    sign >= 55, oldWatermeter/newWatermeter/radiometer >= 50, LCD_SCREEN* >= 60.
    """
    for label in cached.get("scale_labels", []) + cached.get("material_labels", []):
        label_name = label.get("Name", "")
        confidence = label.get("Confidence", 0)
        if label_name.startswith("LCD_SCREEN") and confidence >= 60.0:
            return True
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
    Loop over image files in input_folder, look up their cached metadata in
    output_path/json, feed to Bedrock VLM if a target label is present, parse
    the response, and write text_reading into the JSON cache.
    """
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1] in _IMAGE_EXTENSIONS
    ]
    total = len(image_files)
    current = 0
    local_state = threading.local()
    max_workers = max(1, int(text_config.max_workers))
    pending_jobs: list[tuple[str, str, list]] = []

    for filename in image_files:
        base = os.path.splitext(filename)[0]
        img_path = os.path.join(input_folder, filename)
        cached = common.get_cached_labels(output_path, base)

        if cached.get("text_reading") is not None:
            current += 1
            if progress_callback:
                progress_callback(current, total, f"Skipping {base} (already has text_reading)")
            continue

        if not _has_target_label(cached):
            current += 1
            if progress_callback:
                progress_callback(current, total, f"Skipping {base} (no target labels)")
            continue

        scale_labels = cached.get("scale_labels", [])
        pending_jobs.append((base, img_path, scale_labels))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_base = {
            executor.submit(
                _process_text_reading_worker,
                base,
                img_path,
                scale_labels,
                text_config,
                aws_profile,
                local_state,
            ): base
            for base, img_path, scale_labels in pending_jobs
        }

        for future in as_completed(future_to_base):
            base = future_to_base[future]
            try:
                result = future.result()
                if result.get("error"):
                    print(f"text_reading failed for {base}: {result['error']}")
                else:
                    common.save_cached_labels(
                        output_path,
                        base,
                        text_reading={
                            "digit": result["digit"],
                            "rotation": result["rotation"],
                            "flagged": bool(result["flagged"]),
                        },
                    )
            except Exception as e:
                print(f"text_reading failed for {base}: {e}")

            current += 1
            if progress_callback:
                progress_callback(current, total, f"Processed {current}/{total} images for text_reading")
