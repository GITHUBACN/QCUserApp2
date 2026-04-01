"""
Material classification. No Streamlit.
Uses unified cache: check material_labels before calling API.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend import common

cf_thrsd_dict = {

    "OCC_inventory": 55,
    "OCC_closeup": 55,
    "OCC_scale": 60,
    "OCC_unpacking": 55,
    "MIX_inventory": 70,
    "MIX_closeup": 55,
    "MIX_scale": 70,
    "WHITE_closeup": 55,
    "WHITE_inventory": 55,
    "WHITE_scale": 55,
    "WHITE_unpacking": 55,
    
    # set very high, let model 1 identify the devices
    "newWatermeter": 99,
    "oldWatermeter": 99,
    "radiometer": 99,
    "sign": 99,

    "floor": 55,
  }

class_to_dir = {
    "OCC - inventory": r"classified/1.堆場 4070-10 OCC",
    "OCC - closeup": r"classified/2.貨包 4070-10 OCC",
    "OCC - radiometer - closeup": r"classified/4.輻射儀貨包 4070-10 OCC",
    "OCC - scale": r"classified/5.地面秤重 4070-10 OCC",
    "OCC - newWatermeter": r"classified/7.新款水分儀 4070-10 OCC",
    "OCC - oldWatermeter": r"classified/7.水分儀 4070-10 OCC",
    "OCC - unpacking": r"classified/8.拆包 4070-10 OCC",

    "MIX - inventory": r"classified/1.堆場 4070-30 MIX",
    "MIX - closeup": r"classified/2.貨包 4070-30 MIX",
    "MIX - radiometer - closeup": r"classified/4.輻射儀貨包 4070-30 MIX",
    "MIX - scale": r"classified/5.地面秤重 4070-30 MIX",
    "MIX - newWatermeter": r"classified/7.新款水分儀 4070-30 MIX",
    "MIX - oldWatermeter": r"classified/7.水分儀 4070-30 MIX",
    "MIX - unpacking": r"classified/8.拆包 4070-30 MIX",

    "WHITE - inventory": r"classified/1.堆場 4070-20 WHITE",
    "WHITE - closeup": r"classified/2.貨包 4070-20 WHITE",
    "WHITE - radiometer - closeup": r"classified/4.輻射儀貨包 4070-20 WHITE",
    "WHITE - scale": r"classified/5.地面秤重 4070-20 WHITE",
    "WHITE - newWatermeter": r"classified/7.新款水分儀 4070-20 WHITE",
    "WHITE - oldWatermeter": r"classified/7.水分儀 4070-20 WHITE",
    "WHITE - unpacking": r"classified/8.拆包 4070-20 WHITE",

    "radiometer - floor": r"classified/4. 輻射儀地面",
    "unknown": r"classified/unknown",
}

material_device_translate = {
    "7_MOISTURE": "oldWatermeter",
    "NEW_MOISTURE": "newWatermeter",
    "RADIATION": "radiometer",
    "PAPER_SIGN": "sign",
}

material_specialCase_translate = {
    "MIX_inventory_closeup": "MIX_inventory",
    "MIX_closeup_OCC": "MIX_closeup",
}


def _classify_one(labels: list) -> tuple:
    material_locations = [
        "OCC_inventory", "OCC_closeup", "OCC_scale", "OCC_unpacking",
        "WHITE_inventory", "WHITE_closeup", "WHITE_scale", "WHITE_unpacking",
        "MIX_inventory", "MIX_closeup", "MIX_scale",
    ]
    objects = material_device_translate.keys()
    extras_list = ["floor"]

    pred_material = None
    pred_object = ""
    pred_extra = []
    material_cf = 0
    object_cf = 0

    for label in labels:
        if label["Name"] in material_specialCase_translate:
            label["Name"] = material_specialCase_translate[label["Name"]]
        if (
            label["Name"] in material_locations
            and label["Confidence"] > cf_thrsd_dict.get(label["Name"], 0)
            and label["Confidence"] > material_cf
        ):
            material_cf = label["Confidence"]
            pred_material = label["Name"]
        elif (
            label["Name"] in objects
            and label["Confidence"] > cf_thrsd_dict.get(label["Name"], 0)
            and label["Confidence"] > object_cf
        ):
            object_cf = label["Confidence"]
            pred_object = material_device_translate.get(label["Name"], label["Name"])
        elif (
            label["Name"] in extras_list
            and label["Confidence"] > cf_thrsd_dict.get(label["Name"], 0)
        ):
            pred_extra.append(label["Name"])

    if not pred_material:
        material_cf = 0
        for label in labels:
            if label["Name"] in material_locations and label["Confidence"] > material_cf:
                material_cf = label["Confidence"]
                pred_material = label["Name"]
            if label["Name"] == "floor" and label["Confidence"] > material_cf:
                pred_extra.append("floor")

    return pred_material, pred_object, pred_extra


def _classify_name(pred_material, pred_object: str, pred_extra: list) -> str:
    if not pred_material:
        return "unknown"

    if "Watermeter" in pred_object:
        pred_material = pred_material.split("_")[0]
        return pred_material + " - " + pred_object
    if "radiometer" in pred_object:
        if "floor" in pred_extra:
            return "radiometer - floor"
        pred_material = pred_material.split("_")[0]
        return pred_material + " - " + "radiometer - closeup"
    if "closeup" in pred_material and "sign" in pred_object:
        pred_material = pred_material.split("_")[0]
        return pred_material + " - " + "closeup"
    if "scale" in pred_material and "sign" in pred_object:
        pred_material = pred_material.split("_")[0]
        return pred_material + " - " + "scale"
    if "inventory" in pred_material:
        pred_material = pred_material.split("_")[0]
        return pred_material + " - " + "inventory"
    # labeled unpacking and sign at the same time, mark as closeup
    if "unpacking" in pred_material and "sign" in pred_object:
        pred_material = pred_material.split("_")[0]
        return pred_material + " - " + "closeup"
    pred_material = pred_material.split("_")[0]
    return pred_material + " - " + "unpacking"


def _run_material_inference_worker(
    client,
    model_arn: str,
    uploaded_file: str,
    base: str,
    scale_override: str | None,
) -> dict:
    """Run one material inference and return a structured payload."""
    from PIL import Image

    try:
        with Image.open(uploaded_file) as img:
            image_bytes = common.compress_image(img)
        labels = common.show_custom_labels(
            client, model_arn, image_bytes, min_confidence=10
        )
        pred_material, pred_object, pred_extra = _classify_one(labels)
        if scale_override:
            pred_object = material_device_translate.get(scale_override, pred_object)
        result = _classify_name(pred_material, pred_object, pred_extra)
        return {
            "uploaded_file": uploaded_file,
            "base": base,
            "labels": labels,
            "result": result,
            "error": None,
        }
    except Exception as e:
        return {
            "uploaded_file": uploaded_file,
            "base": base,
            "labels": [],
            "result": None,
            "error": str(e),
        }


def classify_materials(
    client,
    model_arn: str,
    uploaded_files_list: list,
    material_device_list: dict,
    output_path: str,
    progress_callback=None,
    max_workers: int = 4,
) -> None:
    """
    Run material classification. Uses unified cache: if material_labels exist, skip API.
    progress_callback(current, total, message) is optional.
    """
    total = len([
        p for p in uploaded_files_list
        if p[-4:].lower() in (".jpg", ".jpeg", ".png")
    ])
    current = 0
    worker_count = max(1, int(max_workers))
    pending_jobs: list[tuple[str, str, str | None]] = []

    for uploaded_file in uploaded_files_list:
        if uploaded_file[-4:].lower() not in (".jpg", ".jpeg", ".png"):
            continue

        filename = os.path.basename(uploaded_file)
        base = os.path.splitext(filename)[0]
        cached = common.get_cached_labels(output_path, base)

        if cached.get("material_labels"):
            current += 1
            if progress_callback:
                progress_callback(current, total, f"Classifying {current}/{total} files...")
        else:
            pending_jobs.append((uploaded_file, base, material_device_list.get(uploaded_file)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _run_material_inference_worker,
                client,
                model_arn,
                uploaded_file,
                base,
                scale_override,
            ): uploaded_file
            for uploaded_file, base, scale_override in pending_jobs
        }

        for future in as_completed(future_map):
            result_payload = future.result()
            base = result_payload["base"]

            if result_payload["error"]:
                print("Classify Failed:", result_payload["error"])
            else:
                common.save_cached_labels(
                    output_path,
                    base,
                    material_labels=result_payload["labels"],
                    material_class=result_payload["result"],
                )

            current += 1
            if progress_callback:
                progress_callback(current, total, f"Classifying {current}/{total} files...")
