"""
Rekognition client and model control. No Streamlit.
"""
import boto3


def get_rekognition_client(profile_name: str):
    """
    Return boto3 Rekognition client for the given profile.
    Raises with a clear message to run 'aws sso login' on failure.
    """
    try:
        session = boto3.Session(profile_name=profile_name)
        return session.client("rekognition")
    except Exception as e:
        raise RuntimeError(
            f"Failed to create AWS session: {e}. "
            f"Please run 'aws sso login --profile {profile_name}' in your terminal."
        ) from e


def check_status(client, project_arn: str, version_name: str) -> str:
    """Return model status string (e.g. 'RUNNING', 'FAILED')."""
    try:
        describe_response = client.describe_project_versions(
            ProjectArn=project_arn,
            VersionNames=[version_name],
        )
        return describe_response["ProjectVersionDescriptions"][0]["Status"]
    except Exception as e:
        print(e)
        return "FAILED"


def start_model(
    client,
    project_arn: str,
    model_arn: str,
    version_name: str,
    min_inference_units: int = 1,
) -> None:
    """Start the project version and wait until running."""
    try:
        client.start_project_version(
            ProjectVersionArn=model_arn,
            MinInferenceUnits=min_inference_units,
        )
        waiter = client.get_waiter("project_version_running")
        waiter.wait(ProjectArn=project_arn, VersionNames=[version_name])
    except Exception as e:
        print(e)


def stop_model(client, model_arn: str) -> None:
    """Stop the project version."""
    try:
        response = client.stop_project_version(ProjectVersionArn=model_arn)
        print("Status:", response.get("Status", ""))
    except Exception as e:
        print(e)
