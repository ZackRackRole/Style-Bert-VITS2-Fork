import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
import yaml

from style_bert_vits2.logging import logger


def download_jp_models():
    repo_id = "litagin/style_bert_vits2_jvnv"
    files = [
        "jvnv-F1-jp/config.json",
        "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors",
        "jvnv-F1-jp/style_vectors.npy",
        "jvnv-F2-jp/config.json",
        "jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors",
        "jvnv-F2-jp/style_vectors.npy",
        "jvnv-M1-jp/config.json",
        "jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors",
        "jvnv-M1-jp/style_vectors.npy",
        "jvnv-M2-jp/config.json",
        "jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors",
        "jvnv-M2-jp/style_vectors.npy",
    ]

    for file in files:
        local_path = Path("model_assets") / file
        if not local_path.exists():
            logger.info(f"Downloading {file}")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir="model_assets",
            )
        else:
            logger.info(f"Skipping {file}, already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=str, help="Dataset root path (default: Data)", default=None
    )
    parser.add_argument(
        "--assets_root", type=str, help="Assets root path (default: model_assets)", default=None
    )
    args = parser.parse_args()

    download_jp_models()

    # Handle optional path overrides
    default_paths_yml = Path("configs/default_paths.yml")
    paths_yml = Path("configs/paths.yml")
    if not paths_yml.exists():
        shutil.copy(default_paths_yml, paths_yml)

    if args.dataset_root is None and args.assets_root is None:
        return

    with open(paths_yml, encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    if args.assets_root is not None:
        yml_data["assets_root"] = args.assets_root
    if args.dataset_root is not None:
        yml_data["dataset_root"] = args.dataset_root
    with open(paths_yml, "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)


if __name__ == "__main__":
    main()
