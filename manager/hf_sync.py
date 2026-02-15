"""
HF Config Sync — sync configs (loras, user nodes, settings) with HuggingFace Dataset repo.

HF repo structure (e.g. kucher7serg/comfyui-config):
  loras_catalog.json   — LoRA catalog with auto_download flags
  user_nodes.json      — extra custom nodes not baked into Docker
  settings.json        — general settings (profile, comfyui args, etc.)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional


HF_CONFIG_REPO_DEFAULT = "kucher7serg/comfyui-config"


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

    def pull_loras_catalog(self) -> dict:
        """Pull loras_catalog.json from HF."""
        data = self.pull_file("loras_catalog.json")
        if data is None:
            return {"version": "1.0", "loras": []}
        return data

    def push_loras_catalog(self, catalog: dict) -> bool:
        """Push loras_catalog.json to HF."""
        return self.push_file("loras_catalog.json", catalog, "Update LoRA catalog")

    def pull_user_nodes(self) -> dict:
        """Pull user_nodes.json from HF."""
        data = self.pull_file("user_nodes.json")
        if data is None:
            return {"version": "1.0", "nodes": []}
        return data

    def push_user_nodes(self, nodes: dict) -> bool:
        """Push user_nodes.json to HF."""
        return self.push_file("user_nodes.json", nodes, "Update user nodes")

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
