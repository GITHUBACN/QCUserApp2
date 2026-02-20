"""
Material classification. No Streamlit.
Uses unified cache: check material_labels before calling API.
"""
import os

from backend import common

cf_thrsd_dict = {
    "OCC_inventory": 55,
    "OCC_closeup": 55,
    "OCC_scale": 60,
    "OCC_unpacking": 55,
    "DLK_inventory": 60,
    "DLK_closeup": 55,
    "DLK_scale": 70,
    "DLK_unpacking": 55,
    "MIX_inventory": 70,
    "MIX_closeup": 55,
    "MIX_scale": 70,
    "radiometer": 50,
    "newWatermeter": 50,
    "oldWatermeter": 50,
    "floor": 55,
    "sign": 55,
}

class_to_dir = {
    "OCC - inventory": r"classified/OCC/1. 堆場",
    "OCC - closeup": r"classified/OCC/2. 貨包",
    "OCC - radiometer - closeup": r"classified/OCC/4. 輻射儀貨包",
    "OCC - scale": r"classified/OCC/5. 地面秤重",
    "OCC - newWatermeter": r"classified/OCC/7. 新款水分儀",
    "OCC - oldWatermeter": r"classified/OCC/7. 水分儀",
    "OCC - unpacking": r"classified/OCC/8. 拆包",
    "MIX - inventory": r"classified/MIX/1. 堆場",
    "MIX - closeup": r"classified/MIX/2. 貨包",
    "MIX - radiometer - closeup": r"classified/MIX/4. 輻射儀貨包",
    "MIX - scale": r"classified/MIX/5. 地面秤重",
    "MIX - newWatermeter": r"classified/MIX/7. 新款水分儀",
    "MIX - oldWatermeter": r"classified/MIX/7. 水分儀",
    "MIX - unpacking": r"classified/MIX/8. 拆包",
    "radiometer - floor": r"classified/4. 輻射儀地面",
    "unknown": r"classified/unknown",
}

material_device_translate = {
    "7_MOISTURE": "oldWatermeter",
    "NEW_MOISTURE": "newWatermeter",
    "RADIATION": "radiometer",
}


def _classify_one(labels: list) -> tuple:
    material_locations = [
        "OCC_inventory", "OCC_closeup", "OCC_scale", "OCC_unpacking",
        "DLK_inventory", "DLK_closeup", "DLK_scale", "DLK_unpacking",
        "MIX_inventory", "MIX_closeup", "MIX_scale",
    ]
    objects = ["sign", "radiometer", "oldWatermeter", "newWatermeter"]
    extras_list = ["floor"]

    pred_material = None
    pred_object = ""
    pred_extra = []
    material_cf = 0
    object_cf = 0

    for label in labels:
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
            pred_object = label["Name"]
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
    pred_material = pred_material.replace("DLK", "OCC")

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
    if "inventory" in pred_material and "sign" in pred_object:
        pred_material = pred_material.split("_")[0]
        return pred_material + " - " + "inventory"
    pred_material = pred_material.split("_")[0]
    return pred_material + " - " + "unpacking"


def classify_materials(
    client,
    model_arn: str,
    uploaded_files_list: list,
    material_device_list: dict,
    output_path: str,
    progress_callback=None,
) -> None:
    """
    Run material classification. Uses unified cache: if material_labels exist, skip API.
    progress_callback(current, total, message) is optional.
    """
    from PIL import Image

    total = len([
        p for p in uploaded_files_list
        if p[-4:].lower() in (".jpg", ".jpeg", ".png")
    ])
    current = 0

    for uploaded_file in uploaded_files_list:
        if uploaded_file[-4:].lower() not in (".jpg", ".jpeg", ".png"):
            continue

        filename = os.path.basename(uploaded_file)
        base = filename[:-4]
        cached = common.get_cached_labels(output_path, base)

        if cached.get("material_labels"):
            pass  # already classified
        else:
            image_bytes = common.compress_image(Image.open(uploaded_file))
            try:
                labels = common.show_custom_labels(
                    client, model_arn, image_bytes, min_confidence=10
                )
                pred_material, pred_object, pred_extra = _classify_one(labels)
                if uploaded_file in material_device_list:
                    pred_object = material_device_translate.get(
                        material_device_list[uploaded_file], pred_object
                    )
                result = _classify_name(pred_material, pred_object, pred_extra)
                common.save_cached_labels(
                    output_path, base, material_labels=labels, material_class=result
                )
            except Exception as e:
                print("Classify Failed:", e)

        current += 1
        if progress_callback:
            progress_callback(current, total, f"Classifying {current}/{total} files...")
