"""
Scale/device classification. No Streamlit.
Uses unified cache: check scale_labels before calling API.
"""
import os
from PIL import Image, ExifTags
import cv2

from backend import common

class_to_dir = {
    "6_IT_0": r"classified/locations/6 IT",
    "6_BE_0": r"classified/locations/6 BE",
    "6_BE_180": r"classified/locations/6 BE",
    "6_CA_OAK_0": r"classified/locations/6 CA OAK",
    "6_CA_WILMINGTON_0": r"classified/locations/6 CA WILMINGTON",
    "6_CA_WILMINGTON_180": r"classified/locations/6 CA WILMINGTON",
    "6_ES_180": r"classified/locations/6 ES",
    "6_ES_0": r"classified/locations/6 ES",
    "6_FR_0": r"classified/locations/6 FR",
    "6_GA_0": r"classified/locations/6 GA",
    "6_GB_90_CW": r"classified/locations/6 GB",
    "6_GB_0": r"classified/locations/6 GB",
    "6_GR_0": r"classified/locations/6 GR",
    "6_HALIFAX_0": r"classified/locations/6 HALIFAX",
    "6_HALIFAX_180": r"classified/locations/6 HALIFAX",
    "6_HR_0": r"classified/locations/6 HR",
    "6_WA_0": r"classified/locations/6 WA",
    "6_PL_0": r"classified/locations/6 PL",
    "6_NJ_NY_0": r"classified/locations/6 NJ NY",
    "6_VANCOUVER_0": r"classified/locations/6 VANCOUVER",
    "6_NL_0": r"classified/locations/6 NL",
    "6_NEW_SCALES_0": r"classified/locations/6 NEW SCALES",
    "9_WA_0": r"classified/trashLocations/9 WA",
    "9_TW_0": r"classified/trashLocations/9 TW",
    "9_CALIFORNIA_0": r"classified/trashLocations/9 CALIFORNIA",
    "9_EU_0": r"classified/trashLocations/9 EU",
    "9_GA_0": r"classified/trashLocations/9 GA",
    "9_JAPAN_0": r"classified/trashLocations/9 JAPAN",
    "9_KR_0": r"classified/trashLocations/9 KR",
    "9_NJ_NY_0": r"classified/trashLocations/9 NJ NY",
    "9_OAK_0": r"classified/trashLocations/9 OAK",
    "9_OAK_90_CCW": r"classified/trashLocations/9 OAK",
    "9_OAK_180": r"classified/trashLocations/9 OAK",
    "9_VANCOUVER_0": r"classified/trashLocations/9 VANCOUVER",
    "9_NEW_SCALES_0": r"classified/trashLocations/9 NEW SCALES",
    "unknown_device": r"classified/unknown_device",
}

material_devices = ["7_MOISTURE", "NEW_MOISTURE", "RADIATION"]
extras = ["FLOOR", "OCC_PAPER", "MIX_PAPER", "WHITE_PAPER", "NON_PAPER_MATERIAL", "HAND"]


def _smart_fix_orientation(input_path: str) -> Image.Image:
    try:
        with Image.open(input_path) as img:
            exif = img._getexif()
            if exif is None:
                img.load()
                return img.copy()
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            val = exif.get(orientation_key) if orientation_key else None
    except Exception:
        img = Image.open(input_path)
        img.load()
        return img.copy()

    if val == 1 or val is None:
        img = Image.open(input_path)
        img.load()
        return img

    cv_img = cv2.imread(input_path)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    if val == 3:
        fixed = cv2.flip(cv_img, -1)
    elif val == 6:
        fixed = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif val == 8:
        fixed = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        img = Image.open(input_path)
        img.load()
        return img
    return Image.fromarray(fixed)


def _prefix_match(target: str, prefix_list: list) -> bool:
    for i in prefix_list:
        if i in target:
            return True
    return False


def _classify_name(labels: list) -> str:
    max_confidence = 0
    max_label = None
    screen_found = False
    for label in labels:
        if "LCD_SCREEN" in label["Name"]:
            screen_found = True
        elif label["Name"] in extras:
            pass
        elif label["Name"] in class_to_dir or _prefix_match(label["Name"], material_devices):
            if label["Confidence"] > max_confidence:
                max_confidence = label["Confidence"]
                max_label = label["Name"]
    if not max_label:
        return "unknown_device" if screen_found else "next_stage"
    if _prefix_match(max_label, material_devices):
        max_label = next((d for d in material_devices if d in max_label), max_label)
    return max_label


def _save_scale_result(
    output_path: str,
    base: str,
    labels: list,
    result: str,
) -> None:
    """Write scale_labels and scale_class to JSON only. Actual copy runs at pipeline end."""
    common.save_cached_labels(
        output_path, base, scale_labels=labels, scale_class=result
    )


def classify_scales(
    client,
    model_arn: str,
    file_paths: list,
    output_path: str,
    progress_callback=None,
) -> tuple[list, dict]:
    """
    Run scale classification. Returns (toNextStage, material_device_list).
    progress_callback(current: int, total: int, message: str) is optional.
    """
    toNextStage = []
    material_device_list = {}
    total = len([p for p in file_paths if p[-4:].lower() in (".jpg", ".jpeg", ".png")])
    current = 0

    for uploaded_file in file_paths:
        if uploaded_file[-4:].lower() not in (".jpg", ".jpeg", ".png"):
            continue

        filename = os.path.basename(uploaded_file)
        base = filename[:-4]
        cached = common.get_cached_labels(output_path, base)

        if cached.get("scale_labels"):
            labels = cached["scale_labels"]
            result = _classify_name(labels)
            common.save_cached_labels(output_path, base, scale_class=result)
            if result == "next_stage":
                toNextStage.append(uploaded_file)
            elif result in material_devices:
                toNextStage.append(uploaded_file)
                material_device_list[uploaded_file] = result
        else:
            image_pil = _smart_fix_orientation(uploaded_file)
            image_bytes = common.compress_image(image_pil)
            try:
                labels = common.show_custom_labels(
                    client, model_arn, image_bytes, min_confidence=75  # higher confidence for scale classification
                )
                result = _classify_name(labels)
                if result == "next_stage":
                    toNextStage.append(uploaded_file)
                elif result in material_devices:
                    toNextStage.append(uploaded_file)
                    material_device_list[uploaded_file] = result
                _save_scale_result(output_path, base, labels, result)
            except Exception as e:
                print("Classify Failed:", e)

        current += 1
        if progress_callback:
            progress_callback(current, total, f"Classifying {current}/{total} files...")

    return toNextStage, material_device_list
