#!/usr/bin/env python3
"""
ComfyUI Model Manager — Gradio Web UI

Tabs:
  1. Status — GPU info, disk, ComfyUI status
  2. Models — categorized checkboxes, download
  3. LoRAs — catalog with add/download/auto_download
  4. Custom Nodes — install/remove
  5. Add Model — form to add to catalog
  6. Settings — profile, HF token, sync

Usage:
    python3 app.py --port 7860 --catalog config/models_catalog.json --comfyui-path /workspace/ComfyUI
"""

import argparse
import json
import os
import signal
import subprocess
import shutil
import threading
from pathlib import Path

import gradio as gr

try:
    from catalog_manager import CatalogManager, NodesCatalogManager
    from downloader import ModelDownloader
    from hf_sync import HFConfigSync
except ImportError:
    from manager.catalog_manager import CatalogManager, NodesCatalogManager
    from manager.downloader import ModelDownloader
    from manager.hf_sync import HFConfigSync


class ManagerApp:
    def __init__(self, catalog_path: str, comfyui_path: str):
        self.catalog = CatalogManager(catalog_path)
        self.comfyui_path = Path(comfyui_path)

        nodes_path = Path(catalog_path).parent / "nodes_catalog.json"
        self.nodes_catalog = NodesCatalogManager(str(nodes_path))

        self.loras_path = Path(catalog_path).parent / "loras_catalog.json"
        self.loras_catalog = self._load_loras()

        self.downloader = ModelDownloader(str(comfyui_path))
        self.hf_sync = HFConfigSync()
        self.comfyui_process = None

    def _load_loras(self) -> dict:
        if self.loras_path.exists():
            with open(self.loras_path) as f:
                return json.load(f)
        return {"version": "1.0", "loras": []}

    def _save_loras(self):
        with open(self.loras_path, "w") as f:
            json.dump(self.loras_catalog, f, indent=2, ensure_ascii=False)

    # =============== Status Tab ===============

    def get_system_status(self) -> str:
        lines = []

        # GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines.append(f"GPU: {result.stdout.strip()}")
        except Exception:
            lines.append("GPU: not detected")

        # PyTorch
        try:
            import torch
            lines.append(f"PyTorch: {torch.__version__}")
            lines.append(f"CUDA toolkit: {torch.version.cuda}")
            if torch.cuda.is_available():
                lines.append(f"GPU (PyTorch): {torch.cuda.get_device_name(0)}")
                vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
                lines.append(f"VRAM: {vram:.1f}GB")
            else:
                lines.append("CUDA: not available")
        except Exception as e:
            lines.append(f"PyTorch: error ({e})")

        # Disk
        try:
            usage = shutil.disk_usage("/workspace")
            total_gb = usage.total / 1024**3
            free_gb = usage.free / 1024**3
            used_gb = usage.used / 1024**3
            if total_gb > 10000:
                lines.append(f"\nDisk: {used_gb:.1f}GB used (NFS volume)")
            else:
                lines.append(f"\nDisk: {used_gb:.1f}GB / {total_gb:.1f}GB (free: {free_gb:.1f}GB)")
        except Exception:
            pass

        # ComfyUI
        if self.comfyui_process and self.comfyui_process.poll() is None:
            lines.append(f"\nComfyUI: running (PID: {self.comfyui_process.pid})")
        else:
            lines.append("\nComfyUI: not running")

        return "\n".join(lines)

    def start_comfyui(self) -> str:
        if self.comfyui_process and self.comfyui_process.poll() is None:
            return "ComfyUI already running"

        comfyui_args = os.environ.get("COMFYUI_ARGS", "")
        port = os.environ.get("COMFYUI_PORT", "8188")

        cmd = [
            "python3", "main.py",
            "--listen", "0.0.0.0",
            "--port", port,
        ]
        if comfyui_args.strip():
            cmd += comfyui_args.split()

        self.comfyui_process = subprocess.Popen(
            cmd,
            cwd=str(self.comfyui_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return f"ComfyUI started (PID: {self.comfyui_process.pid}, port {port})"

    def stop_comfyui(self) -> str:
        if not self.comfyui_process or self.comfyui_process.poll() is not None:
            return "ComfyUI not running"

        self.comfyui_process.terminate()
        try:
            self.comfyui_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.comfyui_process.kill()

        return "ComfyUI stopped"

    # =============== Models Tab ===============

    def get_model_choices(self) -> dict:
        """Get models organized by category for UI."""
        result = {}
        for cat_id, category in self.catalog.get_categories().items():
            choices = []
            defaults = []
            for m in category.get("models", []):
                label = f"{m['name']} ({m.get('size_gb', 0)}GB)"
                exists = self.catalog.check_model_exists(m, str(self.comfyui_path))
                if exists:
                    label += " [on disk]"
                tier = m.get("tier", 3)
                if tier == 1:
                    label += " [T1]"
                choices.append((label, m["name"]))
                if exists or tier <= 2:
                    defaults.append(m["name"])
            result[cat_id] = {
                "display_name": category.get("display_name", cat_id),
                "choices": choices,
                "defaults": defaults,
            }
        return result

    def download_selected(self, *selected_names_lists) -> str:
        """Download selected models."""
        all_selected = []
        for names in selected_names_lists:
            if names:
                all_selected.extend(names)

        if not all_selected:
            return "Nothing selected"

        models = []
        for name in all_selected:
            m = self.catalog.get_model_by_name(name)
            if m and not self.catalog.check_model_exists(m, str(self.comfyui_path)):
                models.append(m)

        if not models:
            return "All selected models already on disk"

        total_gb = sum(m.get("size_gb", 0) for m in models)
        hf_token = os.environ.get("HF_TOKEN")

        self.downloader.download_models(models, hf_token)
        return f"Downloading: {len(models)} models ({total_gb:.1f}GB)"

    def get_download_log(self) -> str:
        return self.downloader.progress.get_log() or "Log empty"

    # =============== LoRA Tab ===============

    def get_lora_choices(self) -> tuple:
        """Get LoRA choices and defaults for checkbox group."""
        choices = []
        defaults = []
        loras_dir = self.comfyui_path / "models" / "loras"

        for lora in self.loras_catalog.get("loras", []):
            name = lora["name"]
            size = lora.get("size_gb", 0)
            exists = (loras_dir / lora["filename"]).exists()

            label = f"{name} ({size}GB)"
            if exists:
                label += " [on disk]"
            if lora.get("auto_download"):
                label += " [auto]"

            choices.append((label, name))
            if exists or lora.get("auto_download"):
                defaults.append(name)

        return choices, defaults

    def download_selected_loras(self, selected_names: list) -> str:
        """Download selected LoRAs."""
        if not selected_names:
            return "Nothing selected"

        loras_dir = self.comfyui_path / "models" / "loras"
        to_download = []

        for lora in self.loras_catalog.get("loras", []):
            if lora["name"] in selected_names:
                local_path = loras_dir / lora["filename"]
                if not local_path.exists():
                    to_download.append({
                        "name": lora["name"],
                        "filename": lora["filename"],
                        "hf_repo": lora["hf_repo"],
                        "hf_file": lora.get("hf_file", lora["filename"]),
                        "size_gb": lora.get("size_gb", 0),
                        "dest_path": "models/loras",
                        "source": "hf",
                    })

        if not to_download:
            return "All selected LoRAs already on disk"

        total_gb = sum(m.get("size_gb", 0) for m in to_download)
        hf_token = os.environ.get("HF_TOKEN")
        self.downloader.download_models(to_download, hf_token)
        return f"Downloading: {len(to_download)} LoRAs ({total_gb:.1f}GB)"

    def add_lora(self, name: str, hf_repo: str, hf_file: str, size_gb: float, auto_download: bool) -> str:
        """Add a new LoRA to the catalog."""
        if not name or not hf_repo:
            return "Fill required fields: Name, HF Repo"

        filename = hf_file.split("/")[-1] if hf_file else name.replace(" ", "_") + ".safetensors"

        # Check duplicate
        for existing in self.loras_catalog.get("loras", []):
            if existing["filename"] == filename:
                return f"LoRA with filename '{filename}' already exists"

        lora = {
            "name": name,
            "filename": filename,
            "hf_repo": hf_repo,
            "hf_file": hf_file or filename,
            "size_gb": float(size_gb) if size_gb else 0,
            "auto_download": auto_download,
        }

        self.loras_catalog.setdefault("loras", []).append(lora)
        self._save_loras()
        return f"LoRA '{name}' added to catalog"

    def remove_lora(self, name: str) -> str:
        """Remove a LoRA from catalog."""
        loras = self.loras_catalog.get("loras", [])
        for i, lora in enumerate(loras):
            if lora["name"] == name:
                loras.pop(i)
                self._save_loras()
                return f"LoRA '{name}' removed from catalog"
        return f"LoRA '{name}' not found"

    def sync_loras_to_hf(self) -> str:
        """Push loras catalog to HF config repo."""
        if self.hf_sync.push_loras_catalog(self.loras_catalog):
            return "LoRA catalog synced to HuggingFace"
        return "Sync failed (check HF_TOKEN and repo access)"

    def sync_loras_from_hf(self) -> str:
        """Pull loras catalog from HF config repo."""
        data = self.hf_sync.pull_loras_catalog()
        if data and data.get("loras"):
            self.loras_catalog = data
            self._save_loras()
            return f"Pulled {len(data['loras'])} LoRAs from HuggingFace"
        return "No LoRA catalog found on HuggingFace (or empty)"

    # =============== Add Model Tab ===============

    def add_new_model(
        self, name: str, hf_repo: str, hf_file: str,
        category: str, size_gb: float, tags: str
    ) -> str:
        if not name or not hf_repo or not category:
            return "Fill required fields: Name, HF Repo, Category"

        filename = hf_file.split("/")[-1] if hf_file else hf_repo.split("/")[-1] + ".safetensors"
        model = {
            "name": name,
            "filename": filename,
            "size_gb": float(size_gb) if size_gb else 0,
            "tier": 3,
            "source": "hf",
            "hf_repo": hf_repo,
            "hf_file": hf_file or filename,
            "dest_path": f"models/{category}",
            "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
        }

        if self.catalog.add_model(category, model):
            return f"Model '{name}' added to catalog (category: {category})"
        else:
            return f"Model '{name}' already exists (duplicate filename)"

    # =============== Build UI ===============

    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(
            title="ComfyUI Model Manager",
        ) as demo:
            gr.Markdown("# ComfyUI Model Manager")

            with gr.Tab("Status"):
                status_text = gr.Textbox(
                    label="System Info",
                    value=self.get_system_status,
                    lines=12,
                    interactive=False,
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    start_btn = gr.Button("Start ComfyUI", variant="primary")
                    stop_btn = gr.Button("Stop ComfyUI", variant="stop")

                comfyui_status = gr.Textbox(label="ComfyUI", interactive=False)

                refresh_btn.click(self.get_system_status, outputs=status_text)
                start_btn.click(self.start_comfyui, outputs=comfyui_status)
                stop_btn.click(self.stop_comfyui, outputs=comfyui_status)

            with gr.Tab("Models"):
                gr.Markdown("Select models to download. Models marked [on disk] are already present.")

                model_data = self.get_model_choices()
                checkboxes = []

                for cat_id, data in model_data.items():
                    with gr.Accordion(data["display_name"], open=True):
                        cb = gr.CheckboxGroup(
                            choices=data["choices"],
                            value=data["defaults"],
                            label=cat_id,
                        )
                        checkboxes.append(cb)

                if not checkboxes:
                    gr.Markdown("*Model catalog is empty.*")

                download_btn = gr.Button("Download selected", variant="primary", size="lg")
                download_status = gr.Textbox(label="Status", interactive=False)
                download_log = gr.Textbox(label="Download log", lines=15, interactive=False)
                refresh_log_btn = gr.Button("Refresh log")

                if checkboxes:
                    download_btn.click(
                        self.download_selected,
                        inputs=checkboxes,
                        outputs=download_status,
                    )
                refresh_log_btn.click(self.get_download_log, outputs=download_log)

            with gr.Tab("LoRAs"):
                gr.Markdown("### LoRA Catalog")
                gr.Markdown("[auto] = downloads at Pod start. Select and click Download for manual download.")

                lora_choices, lora_defaults = self.get_lora_choices()
                lora_checkboxes = gr.CheckboxGroup(
                    choices=lora_choices,
                    value=lora_defaults,
                    label="Available LoRAs",
                )

                with gr.Row():
                    lora_download_btn = gr.Button("Download selected", variant="primary")
                    lora_sync_push_btn = gr.Button("Save to cloud")
                    lora_sync_pull_btn = gr.Button("Load from cloud")

                lora_status = gr.Textbox(label="Status", interactive=False)

                lora_download_btn.click(
                    self.download_selected_loras,
                    inputs=[lora_checkboxes],
                    outputs=lora_status,
                )
                lora_sync_push_btn.click(self.sync_loras_to_hf, outputs=lora_status)
                lora_sync_pull_btn.click(self.sync_loras_from_hf, outputs=lora_status)

                gr.Markdown("---")
                gr.Markdown("### Add LoRA")

                with gr.Column():
                    lora_name = gr.Textbox(label="Name *", placeholder="My LoRA")
                    lora_hf_repo = gr.Textbox(
                        label="HuggingFace Repo *",
                        placeholder="kucher7serg/my-loras",
                    )
                    lora_hf_file = gr.Textbox(
                        label="Filename in repo",
                        placeholder="lora.safetensors (or path/to/lora.safetensors)",
                    )
                    lora_size = gr.Number(label="Size (GB)", value=0)
                    lora_auto = gr.Checkbox(label="Auto-download at Pod start", value=False)
                    lora_add_btn = gr.Button("Add LoRA", variant="primary")
                    lora_add_result = gr.Textbox(label="Result", interactive=False)

                    lora_add_btn.click(
                        self.add_lora,
                        inputs=[lora_name, lora_hf_repo, lora_hf_file, lora_size, lora_auto],
                        outputs=lora_add_result,
                    )

            with gr.Tab("Custom Nodes"):
                gr.Markdown("### Installed custom nodes")

                nodes = self.nodes_catalog.get_all_nodes()
                if nodes:
                    for node in nodes:
                        installed = self.nodes_catalog.check_node_installed(
                            node, str(self.comfyui_path)
                        )
                        status = "[installed]" if installed else "[not installed]"
                        tier_label = f"Tier {node.get('tier', '?')}"
                        gr.Markdown(
                            f"- **{node['name']}** {status} — {tier_label} — "
                            f"[repo]({node['repo_url']})"
                        )
                else:
                    gr.Markdown("*Node catalog is empty.*")

            with gr.Tab("Add Model"):
                gr.Markdown("### Add model to catalog")
                with gr.Column():
                    add_name = gr.Textbox(label="Model name *", placeholder="Wan Video 1.4B")
                    add_hf_repo = gr.Textbox(
                        label="HuggingFace Repo *",
                        placeholder="wanvideo/wan-1.4b",
                    )
                    add_hf_filename = gr.Textbox(
                        label="Filename in repo",
                        placeholder="model.safetensors",
                    )
                    add_category = gr.Dropdown(
                        choices=[
                            "checkpoints", "loras", "controlnet", "vae",
                            "upscale_models", "clip", "text_encoders",
                            "unet", "diffusion_models", "sams",
                        ],
                        label="Category *",
                    )
                    add_size = gr.Number(label="Size (GB)", value=0)
                    add_tags = gr.Textbox(
                        label="Tags (comma-separated)",
                        placeholder="video, wan, essential",
                    )
                    add_btn = gr.Button("Add", variant="primary")
                    add_result = gr.Textbox(label="Result", interactive=False)

                    add_btn.click(
                        self.add_new_model,
                        inputs=[add_name, add_hf_repo, add_hf_filename,
                                add_category, add_size, add_tags],
                        outputs=add_result,
                    )

            with gr.Tab("Settings"):
                gr.Markdown("### Settings")

                current_profile = os.environ.get("MODEL_PROFILE", "default")
                gr.Textbox(
                    label="Current profile",
                    value=current_profile,
                    interactive=False,
                )

                profiles = self.catalog.get_profiles()
                if profiles:
                    gr.Markdown("**Available profiles:**")
                    for name, data in profiles.items():
                        desc = data.get("description", "")
                        tags = data.get("tags", [])
                        gr.Markdown(f"- **{name}**: {desc} (tags: {', '.join(tags)})")

                gr.Markdown("---")

                hf_token_set = "set" if os.environ.get("HF_TOKEN") else "NOT set"
                gr.Textbox(
                    label="HuggingFace Token",
                    value=f"Status: {hf_token_set}",
                    interactive=False,
                )

                hf_config_repo = os.environ.get("HF_CONFIG_REPO", "kucher7serg/comfyui-config")
                gr.Textbox(
                    label="HF Config Repo",
                    value=hf_config_repo,
                    interactive=False,
                )

                gr.Markdown(
                    "Token and config repo are set via environment variables "
                    "`HF_TOKEN` and `HF_CONFIG_REPO` in RunPod Template."
                )

        return demo


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Model Manager")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--catalog", default="config/models_catalog.json")
    parser.add_argument("--comfyui-path", default="/workspace/ComfyUI")
    args = parser.parse_args()

    app = ManagerApp(
        catalog_path=args.catalog,
        comfyui_path=args.comfyui_path,
    )

    demo = app.build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
