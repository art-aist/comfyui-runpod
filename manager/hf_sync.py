"""
HF Config Sync — sync configs (loras, models, settings) with HuggingFace Dataset repo.

HF repo structure (e.g. kucher7serg/comfyui-config):
  loras_catalog.json       — LoRA catalog
  models_catalog.json      — Models catalog
  comfy.settings.json      — ComfyUI UI settings (auto-synced)
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


HF_CONFIG_REPO_DEFAULT = "kucher7serg/comfyui-config"


def parse_hf_url(url: str) -> dict:
    """Parse a HuggingFace URL into repo_id, filename, and hf_file.

    Supports URLs like:
      https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors
      https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors

    Returns: {"hf_repo": "Comfy-Org/Wan_2.1", "hf_file": "split_files/vae/wan_2.1_vae.safetensors", "filename": "wan_2.1_vae.safetensors"}
    """
    url = url.strip()
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    # Need at least: org/repo/blob_or_resolve/branch/filename
    if len(parts) < 5:
        # Maybe it's just repo_id/filename format
        if len(parts) >= 3:
            repo_id = f"{parts[0]}/{parts[1]}"
            hf_file = "/".join(parts[2:])
            filename = parts[-1]
            return {"hf_repo": repo_id, "hf_file": hf_file, "filename": filename}
        return {"hf_repo": "", "hf_file": "", "filename": ""}

    repo_id = f"{parts[0]}/{parts[1]}"
    # Skip "blob"/"resolve" and branch name (usually "main")
    hf_file = "/".join(parts[4:])
    filename = parts[-1]

    return {"hf_repo": repo_id, "hf_file": hf_file, "filename": filename}


class HFConfigSync:
    """Sync config files to/from a HuggingFace Dataset repo."""

    def __init__(self, hf_repo: str = None, hf_token: str = None, local_dir: str = None):
        self.hf_repo = hf_repo or os.environ.get("HF_CONFIG_REPO", HF_CONFIG_REPO_DEFAULT)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.local_dir = Path(local_dir or "/opt/config/hf_sync")
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def pull_file(self, filename: str) -> Optional[dict]:
        """Download a JSON file from HF repo and return parsed content."""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=filename,
                repo_type="dataset",
                local_dir=str(self.local_dir),
                token=self.hf_token,
            )
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[hf_sync] pull {filename}: {e}")
            return None

    def push_file(self, filename: str, data: dict, commit_message: str = None) -> bool:
        """Upload a JSON file to HF repo."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)

            local_path = self.local_dir / filename
            with open(local_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=commit_message or f"Update {filename}",
            )
            return True
        except Exception as e:
            print(f"[hf_sync] push {filename}: {e}")
            return False

    def pull_raw_file(self, filename: str, dest_path: str) -> bool:
        """Download a file from HF repo to a specific local path (for non-JSON files)."""
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=self.hf_repo,
                filename=filename,
                repo_type="dataset",
                local_dir=str(self.local_dir),
                token=self.hf_token,
            )
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, dest)
            return True
        except Exception as e:
            print(f"[hf_sync] pull_raw {filename}: {e}")
            return False

    def push_raw_file(self, local_path: str, path_in_repo: str, commit_message: str = None) -> bool:
        """Upload any file from local path to HF repo."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=commit_message or f"Update {path_in_repo}",
            )
            return True
        except Exception as e:
            print(f"[hf_sync] push_raw {path_in_repo}: {e}")
            return False

    def pull_loras_catalog(self) -> dict:
        """Pull loras_catalog.json from HF."""
        data = self.pull_file("loras_catalog.json")
        if data is None:
            return {"version": "1.0", "loras": []}
        return data

    def push_loras_catalog(self, catalog: dict) -> bool:
        """Push loras_catalog.json to HF."""
        return self.push_file("loras_catalog.json", catalog, "Update LoRA catalog")

    def pull_models_catalog(self) -> Optional[dict]:
        """Pull models_catalog.json from HF."""
        return self.pull_file("models_catalog.json")

    def push_models_catalog(self, catalog: dict) -> bool:
        """Push models_catalog.json to HF."""
        return self.push_file("models_catalog.json", catalog, "Update models catalog")

    def pull_nodes_catalog(self) -> Optional[dict]:
        """Pull nodes_catalog.json from HF."""
        return self.pull_file("nodes_catalog.json")

    def push_nodes_catalog(self, catalog: dict) -> bool:
        """Push nodes_catalog.json to HF."""
        return self.push_file("nodes_catalog.json", catalog, "Update nodes catalog")

    def pull_comfy_settings(self, comfyui_path: str) -> bool:
        """Pull comfy.settings.json from HF and place it in ComfyUI user dir."""
        dest = f"{comfyui_path}/user/default/comfy.settings.json"
        return self.pull_raw_file("comfy.settings.json", dest)

    def push_comfy_settings(self, comfyui_path: str) -> bool:
        """Push comfy.settings.json from ComfyUI to HF."""
        src = f"{comfyui_path}/user/default/comfy.settings.json"
        if not os.path.exists(src):
            print("[hf_sync] comfy.settings.json not found, skipping push")
            return False
        return self.push_raw_file(src, "comfy.settings.json", "Update ComfyUI settings")

    def ensure_repo_exists(self) -> bool:
        """Create the HF Dataset repo if it doesn't exist."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            api.create_repo(
                repo_id=self.hf_repo,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            return True
        except Exception as e:
            print(f"[hf_sync] create repo: {e}")
            return False
