"""
All Streamlit UI: folder selection, model status, button, progress, messages.
Calls backend and config; no classification or AWS logic.
"""
import os
import streamlit as st

from backend import aws, common, scales, materials


def show_connection_status(connected: bool, profile: str) -> None:
    if connected:
        st.success(f"Connected using local profile: {profile}")
    else:
        st.error(f"Please run 'aws sso login --profile {profile}' in your terminal.")


def render_folder_selection() -> tuple[list[str], str, str]:
    """Returns (files, folder_input, folder_output). Uses st.text_input, st.write, st.error."""
    st.write("Select Folders")
    files = []
    folder_input = ""
    folder_output = ""

    folder_input = st.text_input("Input folder")
    if folder_input and os.path.exists(folder_input):
        files = [os.path.join(folder_input, f) for f in os.listdir(folder_input)]
        st.write(f"Found {len(files)} files in {folder_input}")
    elif folder_input:
        st.error(f"Folder {folder_input} does not exist.")

    folder_output = st.text_input("Output folder")
    if folder_output and os.path.exists(folder_output):
        st.write(f"Files will be saved to {folder_output}")
    elif folder_output:
        st.error(f"Folder {folder_output} does not exist.")

    return files, folder_input, folder_output


def render_model_status(status1: str, status2: str) -> None:
    st.space("medium")
    st.write(f"Scale Model status: {status1}")
    st.write(f"Material Model status: {status2}")


def render_start_button(disabled: bool) -> bool:
    """Returns True if the button was clicked."""
    return bool(st.button("Start Classifying", type="primary", disabled=disabled))


def run_classification_pipeline(config, client, files: list, folder_input: str, folder_output: str) -> None:
    """
    Run scales then materials with progress callbacks, then text reading.
    Show st.success when done.
    """
    progress_placeholder = st.empty()
    message_placeholder = st.empty()

    def progress_callback(current: int, total: int, message: str):
        if total > 0:
            progress_placeholder.progress(current / total)
        message_placeholder.write(message)

    toNextStage, material_device_list = scales.classify_scales(
        client,
        config.model1_arn,
        files,
        folder_output,
        progress_callback=progress_callback,
    )

    materials.classify_materials(
        client,
        config.model2_arn,
        toNextStage,
        material_device_list,
        folder_output,
        progress_callback=progress_callback,
    )

    # Run text reading after classification completes
    try:
        from config import get_text_reading_config
        from backend.text_reading import add_text_reading_to_jsons

        text_config = get_text_reading_config()
        message_placeholder.write("Running text reading on images with target labels...")
        add_text_reading_to_jsons(
            input_folder=folder_input,
            output_path=folder_output,
            text_config=text_config,
            aws_profile=config.aws_profile,
            progress_callback=progress_callback,
        )
    except Exception as e:
        # Text reading is optional; log but don't fail the pipeline
        st.warning(f"Text reading skipped: {e}")

    # Copy images to classified folders once at the end (scales and materials write only to JSON)
    message_placeholder.write("Copying images to classified folders...")
    common.copy_images_to_classified_folders(
        folder_output,
        folder_input,
        material_class_to_dir=materials.class_to_dir,
        scale_class_to_dir=scales.class_to_dir,
        progress_callback=progress_callback,
    )

    progress_placeholder.progress(1.0)
    message_placeholder.empty()
    st.success("Finish")
