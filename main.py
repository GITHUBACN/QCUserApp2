"""
QC User App: orchestration only. Load config, wire display and backend.
"""
import os

from config import get_config
from backend import aws
from display import streamlit_ui as ui


def main() -> None:
    config = get_config()

    try:
        client = aws.get_rekognition_client(config.aws_profile)
        ui.show_connection_status(True, config.aws_profile)
    except RuntimeError:
        ui.show_connection_status(False, config.aws_profile)
        client = None

    files, folder_input, folder_output = ui.render_folder_selection()

    if client is not None:
        model1_status = aws.check_status(
            client, config.project1_arn, config.version_name1
        )
        model2_status = aws.check_status(
            client, config.project2_arn, config.version_name2
        )
    else:
        model1_status = model2_status = "FAILED"

    ui.render_model_status(model1_status, model2_status)

    can_start = (
        len(files) > 0
        and folder_output
        and os.path.exists(folder_output)
        and model1_status == "RUNNING"
        and model2_status == "RUNNING"
    )
    if ui.render_start_button(disabled=not can_start) and client is not None:
        ui.run_classification_pipeline(config, client, files, folder_input, folder_output)


if __name__ == "__main__":
    main()
