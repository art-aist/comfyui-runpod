"""
Catalog Manager — load/save model and node catalogs.
"""

import json
import os
from pathlib import Path
from typing import Optional


class CatalogManager:
    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self.catalog = self._load()

    def _load(self) -> dict:
        if self.catalog_path.exists():
            with open(self.catalog_path) as f:
                return json.load(f)
        return {"version": "1.0", "categories": {}, "profiles": {}}

    def save(self):
        with open(self.catalog_path, "w") as f:
            json.dump(self.catalog, f, indent=2, ensure_ascii=False)

    def get_categories(self) -> dict:
        return self.catalog.get("categories", {})

    def get_profiles(self) -> dict:
        return self.catalog.get("profiles", {})

    def get_all_models(self) -> list:
        models = []
        for cat_name, cat in self.catalog.get("categories", {}).items():
            for m in cat.get("models", []):
                m_copy = dict(m)
                m_copy["_category"] = cat_name
                models.append(m_copy)
        return models

    def get_model_by_name(self, name: str) -> Optional[dict]:
        for m in self.get_all_models():
            if m["name"] == name:
                return m
        return None

    def add_model(self, category: str, model: dict) -> bool:
        """Add a new model to the catalog."""
        categories = self.catalog.setdefault("categories", {})

        if category not in categories:
            categories[category] = {
                "display_name": category.replace("_", " ").title(),
                "models": [],
            }

        # Check for duplicate by filename
        for existing in categories[category]["models"]:
            if existing["filename"] == model["filename"]:
                return False

        categories[category]["models"].append(model)
        self.save()
        return True

    def remove_model(self, name: str) -> bool:
        """Remove a model from the catalog by name."""
        for cat in self.catalog.get("categories", {}).values():
            models = cat.get("models", [])
            for i, m in enumerate(models):
                if m["name"] == name:
                    models.pop(i)
                    self.save()
                    return True
        return False

    def get_models_for_profile(self, profile_name: str, tier_filter: list = None) -> list:
        """Get models matching profile tags and optional tier filter."""
        profile = self.get_profiles().get(profile_name, {})
        profile_tags = set(profile.get("tags", []))

        selected = []
        for m in self.get_all_models():
            if tier_filter and m.get("tier") not in tier_filter:
                continue
            model_tags = set(m.get("tags", []))
            if "*" in profile_tags or model_tags & profile_tags:
                selected.append(m)
        return selected

    def check_model_exists(self, model: dict, comfyui_path: str) -> bool:
        """Check if model file already exists on disk."""
        local_path = Path(comfyui_path) / model["dest_path"] / model["filename"]
        return local_path.exists()

    def get_model_local_path(self, model: dict, comfyui_path: str) -> Path:
        return Path(comfyui_path) / model["dest_path"] / model["filename"]

    def update_selected(self, selected_names: list):
        """Update 'selected' field on all models based on checkbox state."""
        selected_set = set(selected_names or [])
        for cat in self.catalog.get("categories", {}).values():
            for m in cat.get("models", []):
                m["selected"] = m["name"] in selected_set
        self.save()


class NodesCatalogManager:
    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self.catalog = self._load()

    def _load(self) -> dict:
        if self.catalog_path.exists():
            with open(self.catalog_path) as f:
                return json.load(f)
        return {"version": "1.0", "nodes": []}

    def save(self):
        with open(self.catalog_path, "w") as f:
            json.dump(self.catalog, f, indent=2, ensure_ascii=False)

    def get_nodes(self, max_tier: int = 3) -> list:
        return [n for n in self.catalog.get("nodes", []) if n.get("tier", 3) <= max_tier]

    def get_all_nodes(self) -> list:
        return self.catalog.get("nodes", [])

    def check_node_installed(self, node: dict, comfyui_path: str) -> bool:
        """Check if a custom node is already installed."""
        repo_url = node.get("repo_url", "")
        if not repo_url:
            return False
        name = os.path.basename(repo_url).replace(".git", "")
        return os.path.exists(os.path.join(comfyui_path, "custom_nodes", name))

    def add_node(self, name: str, repo_url: str, tier: int = 2) -> bool:
        """Add a node to the catalog. Returns False if duplicate."""
        for existing in self.catalog.get("nodes", []):
            if existing.get("repo_url", "").rstrip("/") == repo_url.rstrip("/"):
                return False

        self.catalog.setdefault("nodes", []).append({
            "name": name,
            "repo_url": repo_url,
            "tier": tier,
        })
        self.save()
        return True

    def remove_node(self, repo_url: str) -> bool:
        """Remove a node from the catalog by repo_url. Returns False if not found."""
        nodes = self.catalog.get("nodes", [])
        for i, node in enumerate(nodes):
            if node.get("repo_url", "").rstrip("/") == repo_url.rstrip("/"):
                nodes.pop(i)
                self.save()
                return True
        return False

    def remove_node_by_name(self, folder_name: str) -> bool:
        """Remove a node from the catalog by folder name (basename of repo_url)."""
        nodes = self.catalog.get("nodes", [])
        for i, node in enumerate(nodes):
            node_folder = os.path.basename(node.get("repo_url", "")).replace(".git", "")
            if node_folder == folder_name:
                nodes.pop(i)
                self.save()
                return True
        return False
