#!/usr/bin/env python3
"""
ComfyUI Model Manager — Gradio Web UI

Tabs:
  1. Status — GPU info, disk, ComfyUI status
  2. Models — list with checkboxes + simple add form
  3. LoRAs — list with checkboxes + simple add form
  4. Custom Nodes — installed nodes list
  5. Settings — profile, HF token info

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
import time
from pathlib import Path

import gradio as gr

try:
    from catalog_manager import CatalogManager, NodesCatalogManager
    from downloader import ModelDownloader
    from hf_sync import HFConfigSync, parse_hf_url
except ImportError:
    from manager.catalog_manager import CatalogManager, NodesCatalogManager
    from manager.downloader import ModelDownloader
    from manager.hf_sync import HFConfigSync, parse_hf_url


class SettingsWatcher:
    """Watches comfy.settings.json and auto-pushes to HF every hour or on shutdown."""

    def __init__(self, comfyui_path: str, hf_sync: HFConfigSync, interval: int = 3600):
        self.comfyui_path = comfyui_path
        self.hf_sync = hf_sync
        self.interval = interval  # seconds between checks (default: 1 hour)
        self.settings_path = Path(comfyui_path) / "user" / "default" / "comfy.settings.json"
        self._last_mtime = self._get_mtime()
        self._stop_event = threading.Event()
        self._thread = None

    def _get_mtime(self) -> float:
        try:
            return self.settings_path.stat().st_mtime
        except FileNotFoundError:
            return 0

    def _has_changed(self) -> bool:
        current_mtime = self._get_mtime()
        if current_mtime > self._last_mtime:
            self._last_mtime = current_mtime
            return True
        return False

    def sync_now(self):
        """Push settings to HF if file exists."""
        if self.settings_path.exists():
            if self.hf_sync.push_comfy_settings(self.comfyui_path):
                print("[settings_watcher] Synced comfy.settings.json to HF")
            else:
                print("[settings_watcher] Failed to sync settings")

    def _worker(self):
        while not self._stop_event.wait(self.interval):
            if self._has_changed():
                self.sync_now()

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[settings_watcher] Started (interval: {self.interval}s)")

    def stop(self):
        self._stop_event.set()
        # Final sync on shutdown
        if self._has_changed():
            self.sync_now()


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

        # Start settings watcher
        self.settings_watcher = SettingsWatcher(str(comfyui_path), self.hf_sync)
        self.settings_watcher.start()

        # Register shutdown handler
        signal.signal(signal.SIGTERM, self._on_shutdown)
        signal.signal(signal.SIGINT, self._on_shutdown)

    def _on_shutdown(self, signum, frame):
        print(f"[manager] Shutdown signal ({signum}), syncing settings...")
        self.settings_watcher.stop()
        raise SystemExit(0)

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
        """Get models organized by category for UI. Checked state from catalog 'selected' field."""
        result = {}
        for cat_id, category in self.catalog.get_categories().items():
            choices = []
            defaults = []
            for m in category.get("models", []):
                label = f"{m['name']} ({m.get('size_gb', 0)}GB)"
                exists = self.catalog.check_model_exists(m, str(self.comfyui_path))
                if exists:
                    label += " [on disk]"
                choices.append((label, m["name"]))
                if m.get("selected", False) or exists:
                    defaults.append(m["name"])
            result[cat_id] = {
                "display_name": category.get("display_name", cat_id),
                "choices": choices,
                "defaults": defaults,
            }
        return result

    def download_selected(self, *selected_names_lists) -> str:
        """Download selected models and save selection state."""
        all_selected = []
        for names in selected_names_lists:
            if names:
                all_selected.extend(names)

        # Save selection state to catalog (persist across sessions)
        self.catalog.update_selected(all_selected)
        self.hf_sync.push_models_catalog(self.catalog.catalog)

        if not all_selected:
            return "Selection saved. Nothing to download."

        models = []
        for name in all_selected:
            m = self.catalog.get_model_by_name(name)
            if m and not self.catalog.check_model_exists(m, str(self.comfyui_path)):
                models.append(m)

        if not models:
            return "Selection saved. All selected models already on disk."

        total_gb = sum(m.get("size_gb", 0) for m in models)
        hf_token = os.environ.get("HF_TOKEN")

        self.downloader.download_models(models, hf_token)
        return f"Downloading: {len(models)} models ({total_gb:.1f}GB)"

    def get_download_log(self) -> str:
        return self.downloader.progress.get_log() or "Log empty"

    def add_new_model(self, name: str, folder: str, hf_link: str) -> str:
        """Add a new model to the catalog. Parses HF URL automatically."""
        if not name or not folder or not hf_link:
            return "Fill all fields: Name, Folder, HF Link"

        parsed = parse_hf_url(hf_link)
        if not parsed["hf_repo"]:
            return "Could not parse HF link. Use format: https://huggingface.co/org/repo/blob/main/file.safetensors"

        model = {
            "name": name,
            "filename": parsed["filename"],
            "size_gb": 0,
            "tier": 3,
            "source": "hf",
            "hf_repo": parsed["hf_repo"],
            "hf_file": parsed["hf_file"],
            "dest_path": f"models/{folder}/",
            "tags": [],
        }

        if self.catalog.add_model(folder, model):
            # Sync catalog to HF
            self.hf_sync.push_models_catalog(self.catalog.catalog)
            return f"Model '{name}' added (folder: {folder})"
        else:
            return f"Model '{name}' already exists (duplicate filename)"

    # =============== LoRA Tab ===============

    def get_lora_choices(self) -> tuple:
        """Get LoRA choices and defaults. Checked state from 'selected' field."""
        choices = []
        defaults = []
        loras_dir = self.comfyui_path / "models" / "loras"

        for lora in self.loras_catalog.get("loras", []):
            name = lora["name"]
            exists = (loras_dir / lora["filename"]).exists()

            label = name
            if exists:
                label += " [on disk]"

            choices.append((label, name))
            if lora.get("selected", False) or exists:
                defaults.append(name)

        return choices, defaults

    def download_selected_loras(self, selected_names: list) -> str:
        """Download selected LoRAs and save selection state."""
        # Save selection state
        selected_set = set(selected_names or [])
        for lora in self.loras_catalog.get("loras", []):
            lora["selected"] = lora["name"] in selected_set
        self._save_loras()
        self.hf_sync.push_loras_catalog(self.loras_catalog)

        if not selected_names:
            return "Selection saved. Nothing to download."

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
            return "Selection saved. All selected LoRAs already on disk."

        total_gb = sum(m.get("size_gb", 0) for m in to_download)
        hf_token = os.environ.get("HF_TOKEN")
        self.downloader.download_models(to_download, hf_token)
        return f"Downloading: {len(to_download)} LoRAs ({total_gb:.1f}GB)"

    def add_lora(self, name: str, hf_link: str) -> str:
        """Add a new LoRA to the catalog. Only 2 fields: name + HF link."""
        if not name or not hf_link:
            return "Fill both fields: Name and HF Link"

        parsed = parse_hf_url(hf_link)
        if not parsed["hf_repo"]:
            return "Could not parse HF link. Use format: https://huggingface.co/org/repo/blob/main/lora.safetensors"

        # Check duplicate
        for existing in self.loras_catalog.get("loras", []):
            if existing["filename"] == parsed["filename"]:
                return f"LoRA with filename '{parsed['filename']}' already exists"

        lora = {
            "name": name,
            "filename": parsed["filename"],
            "hf_repo": parsed["hf_repo"],
            "hf_file": parsed["hf_file"],
            "size_gb": 0,
        }

        self.loras_catalog.setdefault("loras", []).append(lora)
        self._save_loras()
        # Sync to HF
        self.hf_sync.push_loras_catalog(self.loras_catalog)
        return f"LoRA '{name}' added to catalog"

    def remove_lora(self, name: str) -> str:
        """Remove a LoRA from catalog."""
        loras = self.loras_catalog.get("loras", [])
        for i, lora in enumerate(loras):
            if lora["name"] == name:
                loras.pop(i)
                self._save_loras()
                self.hf_sync.push_loras_catalog(self.loras_catalog)
                return f"LoRA '{name}' removed from catalog"
        return f"LoRA '{name}' not found"

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
                gr.Markdown("Select models and click Download. Models marked [on disk] are already present.")

                model_data = self.get_model_choices()
                checkboxes = []

                for cat_id, data in model_data.items():
                    if data["choices"]:
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

                gr.Markdown("---")
                gr.Markdown("### Add Model")

                with gr.Column():
                    model_name = gr.Textbox(label="Name", placeholder="Wan Video 14B fp16")
                    model_folder = gr.Dropdown(
                        choices=[
                            "checkpoints", "unet", "diffusion_models",
                            "clip", "clip_vision", "text_encoders",
                            "vae", "controlnet", "upscale_models", "sams",
                        ],
                        label="Folder",
                    )
                    model_hf_link = gr.Textbox(
                        label="HuggingFace Link",
                        placeholder="https://huggingface.co/org/repo/blob/main/model.safetensors",
                    )
                    model_add_btn = gr.Button("Add Model", variant="primary")
                    model_add_result = gr.Textbox(label="Result", interactive=False)

                    model_add_btn.click(
                        self.add_new_model,
                        inputs=[model_name, model_folder, model_hf_link],
                        outputs=model_add_result,
                    )

            with gr.Tab("LoRAs"):
                gr.Markdown("Select LoRAs and click Download.")

                lora_choices, lora_defaults = self.get_lora_choices()
                lora_checkboxes = gr.CheckboxGroup(
                    choices=lora_choices,
                    value=lora_defaults,
                    label="Available LoRAs",
                )

                lora_download_btn = gr.Button("Download selected", variant="primary")
                lora_status = gr.Textbox(label="Status", interactive=False)

                lora_download_btn.click(
                    self.download_selected_loras,
                    inputs=[lora_checkboxes],
                    outputs=lora_status,
                )

                gr.Markdown("---")
                gr.Markdown("### Add LoRA")

                with gr.Column():
                    lora_name = gr.Textbox(label="Name", placeholder="My LoRA")
                    lora_hf_link = gr.Textbox(
                        label="HuggingFace Link",
                        placeholder="https://huggingface.co/org/repo/blob/main/lora.safetensors",
                    )
                    lora_add_btn = gr.Button("Add LoRA", variant="primary")
                    lora_add_result = gr.Textbox(label="Result", interactive=False)

                    lora_add_btn.click(
                        self.add_lora,
                        inputs=[lora_name, lora_hf_link],
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
                    "ComfyUI settings auto-sync to HuggingFace every hour and on Pod shutdown.\n\n"
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
