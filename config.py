"""
Load configuration from environment. Use get_config() so the rest of the app
does not read os.environ directly.
"""
import os
from typing import NamedTuple

def _load_env_file() -> None:
    _config_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(_config_dir, ".env")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass
    # Fallback: read .env manually
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


_load_env_file()


class Config(NamedTuple):
    aws_profile: str
    project1_arn: str
    project2_arn: str
    model1_arn: str
    model2_arn: str
    version_name1: str
    version_name2: str
    min_inference_units: int


_REQUIRED = (
    "AWS_PROFILE",
    "PROJECT1_ARN",
    "PROJECT2_ARN",
    "MODEL1_ARN",
    "MODEL2_ARN",
    "VERSION_NAME1",
    "VERSION_NAME2",
)


def get_config() -> Config:
    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and set values, or export them."
        )
    return Config(
        aws_profile=os.getenv("AWS_PROFILE", "").strip(),
        project1_arn=os.getenv("PROJECT1_ARN", "").strip(),
        project2_arn=os.getenv("PROJECT2_ARN", "").strip(),
        model1_arn=os.getenv("MODEL1_ARN", "").strip(),
        model2_arn=os.getenv("MODEL2_ARN", "").strip(),
        version_name1=os.getenv("VERSION_NAME1", "").strip(),
        version_name2=os.getenv("VERSION_NAME2", "").strip(),
        min_inference_units=int(os.getenv("MIN_INFERENCE_UNITS", "1")),
    )


class TextReadingConfig(NamedTuple):
    prompt: str
    model_id: str
    region: str


def _load_text_reading_prompt() -> str:
    """
    Load the text-reading prompt from either a file or an env var.

    - If TEXT_READING_PROMPT_FILE is set, read the file.
    - Else use TEXT_READING_PROMPT (may contain \n escapes).
    - Fallback: a minimal default prompt if both are empty.
    """
    prompt_file = os.getenv("TEXT_READING_PROMPT_FILE")
    if prompt_file:
        # If relative, resolve relative to this config file directory.
        if not os.path.isabs(prompt_file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_file = os.path.join(base_dir, prompt_file)
        if os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()

    prompt = os.getenv("TEXT_READING_PROMPT", "").strip()
    if prompt:
        # Allow \n sequences for multi-line prompts
        return prompt.replace("\\n", "\n")

    # Minimal safe default that matches the notebook contract.
    return (
        "You are reading digits from photos of paper signs and device screens.\n"
        "Return a brief reasoning, then on the last line output \"{digit/HSCODE} - {flagged/None}\"."
    )


def get_text_reading_config() -> TextReadingConfig:
    """
    Return configuration for the text-reading backend.
    Does not enforce presence of these env vars; uses sensible defaults.
    """
    prompt = _load_text_reading_prompt()
    model_id = os.getenv("BEDROCK_MODEL_ID", "us.meta.llama3-2-90b-instruct-v1:0").strip()
    region = os.getenv("BEDROCK_REGION", "us-east-2").strip()
    return TextReadingConfig(prompt=prompt, model_id=model_id, region=region)
