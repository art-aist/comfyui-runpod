#!/usr/bin/env python3
"""
ComfyUI Model Manager — Gradio Web UI

Tabs:
  1. Status — GPU info, disk, ComfyUI status
  2. Models — categorized checkboxes, download
  3. Custom Nodes — install/remove
  4. Add Model — form to add to catalog
  5. Settings — profile, HF token

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
except ImportError:
    from manager.catalog_manager import CatalogManager, NodesCatalogManager
    from manager.downloader import ModelDownloader


class ManagerApp:
    def __init__(self, catalog_path: str, comfyui_path: str):
        self.catalog = CatalogManager(catalog_path)
        self.comfyui_path = Path(comfyui_path)

        nodes_path = Path(catalog_path).parent / "nodes_catalog.json"
        self.nodes_catalog = NodesCatalogManager(str(nodes_path))

        self.downloader = ModelDownloader(str(comfyui_path))
        self.comfyui_process = None

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
            lines.append("GPU: не определён")

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
                lines.append("CUDA: не доступен")
        except Exception as e:
            lines.append(f"PyTorch: ошибка ({e})")

        # Disk
        try:
            usage = shutil.disk_usage("/workspace")
            total_gb = usage.total / 1024**3
            free_gb = usage.free / 1024**3
            used_gb = usage.used / 1024**3
            lines.append(f"\nДиск: {used_gb:.1f}GB / {total_gb:.1f}GB (свободно: {free_gb:.1f}GB)")
        except Exception:
            pass

        # ComfyUI
        if self.comfyui_process and self.comfyui_process.poll() is None:
            lines.append(f"\nComfyUI: запущен (PID: {self.comfyui_process.pid})")
        else:
            lines.append("\nComfyUI: не запущен")

        return "\n".join(lines)

    def start_comfyui(self) -> str:
        if self.comfyui_process and self.comfyui_process.poll() is None:
            return "ComfyUI уже запущен"

        comfyui_args = os.environ.get("COMFYUI_ARGS", "--highvram")
        port = os.environ.get("COMFYUI_PORT", "8188")

        cmd = [
            "python3", "main.py",
            "--listen", "0.0.0.0",
            "--port", port,
        ] + comfyui_args.split()

        self.comfyui_process = subprocess.Popen(
            cmd,
            cwd=str(self.comfyui_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return f"ComfyUI запущен (PID: {self.comfyui_process.pid}, порт {port})"

    def stop_comfyui(self) -> str:
        if not self.comfyui_process or self.comfyui_process.poll() is not None:
            return "ComfyUI не запущен"

        self.comfyui_process.terminate()
        try:
            self.comfyui_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.comfyui_process.kill()

        return "ComfyUI остановлен"

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
                    label += " [на диске]"
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
            return "Ничего не выбрано"

        models = []
        for name in all_selected:
            m = self.catalog.get_model_by_name(name)
            if m and not self.catalog.check_model_exists(m, str(self.comfyui_path)):
                models.append(m)

        if not models:
            return "Все выбранные модели уже на диске"

        total_gb = sum(m.get("size_gb", 0) for m in models)
        hf_token = os.environ.get("HF_TOKEN")

        self.downloader.download_models(models, hf_token)
        return f"Начало скачивания: {len(models)} моделей ({total_gb:.1f}GB)"

    def get_download_log(self) -> str:
        return self.downloader.progress.get_log() or "Лог пуст"

    # =============== Add Model Tab ===============

    def add_new_model(
        self, name: str, hf_repo: str, hf_file: str,
        category: str, size_gb: float, tags: str
    ) -> str:
        if not name or not hf_repo or not category:
            return "Заполните обязательные поля: Name, HF Repo, Category"

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
            return f"Модель '{name}' добавлена в каталог (категория: {category})"
        else:
            return f"Модель '{name}' уже существует (дубликат по filename)"

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
                    refresh_btn = gr.Button("Обновить")
                    start_btn = gr.Button("Запустить ComfyUI", variant="primary")
                    stop_btn = gr.Button("Остановить ComfyUI", variant="stop")

                comfyui_status = gr.Textbox(label="ComfyUI", interactive=False)

                refresh_btn.click(self.get_system_status, outputs=status_text)
                start_btn.click(self.start_comfyui, outputs=comfyui_status)
                stop_btn.click(self.stop_comfyui, outputs=comfyui_status)

            with gr.Tab("Models"):
                gr.Markdown("Выберите модели для скачивания. Модели с пометкой [на диске] уже есть.")

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
                    gr.Markdown("*Каталог моделей пуст. Добавьте модели на вкладке 'Add Model' или выполните инвентаризацию.*")

                download_btn = gr.Button("Скачать выбранные", variant="primary", size="lg")
                download_status = gr.Textbox(label="Статус", interactive=False)
                download_log = gr.Textbox(label="Лог скачивания", lines=15, interactive=False)
                refresh_log_btn = gr.Button("Обновить лог")

                if checkboxes:
                    download_btn.click(
                        self.download_selected,
                        inputs=checkboxes,
                        outputs=download_status,
                    )
                refresh_log_btn.click(self.get_download_log, outputs=download_log)

            with gr.Tab("Custom Nodes"):
                gr.Markdown("### Установленные custom nodes")

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
                    gr.Markdown("*Каталог nodes пуст. Выполните инвентаризацию.*")

            with gr.Tab("Add Model"):
                gr.Markdown("### Добавить модель в каталог")
                with gr.Column():
                    add_name = gr.Textbox(label="Название модели *", placeholder="Wan Video 1.4B")
                    add_hf_repo = gr.Textbox(
                        label="HuggingFace Repo *",
                        placeholder="wanvideo/wan-1.4b",
                    )
                    add_hf_filename = gr.Textbox(
                        label="Filename в репо",
                        placeholder="model.safetensors",
                    )
                    add_category = gr.Dropdown(
                        choices=[
                            "checkpoints", "loras", "controlnet", "vae",
                            "upscale_models", "clip", "text_encoders",
                            "unet", "diffusion_models",
                        ],
                        label="Категория *",
                    )
                    add_size = gr.Number(label="Размер (GB)", value=0)
                    add_tags = gr.Textbox(
                        label="Теги (через запятую)",
                        placeholder="video, wan, essential",
                    )
                    add_btn = gr.Button("Добавить", variant="primary")
                    add_result = gr.Textbox(label="Результат", interactive=False)

                    add_btn.click(
                        self.add_new_model,
                        inputs=[add_name, add_hf_repo, add_hf_filename,
                                add_category, add_size, add_tags],
                        outputs=add_result,
                    )

            with gr.Tab("Settings"):
                gr.Markdown("### Настройки")

                current_profile = os.environ.get("MODEL_PROFILE", "default")
                gr.Textbox(
                    label="Текущий профиль",
                    value=current_profile,
                    interactive=False,
                )

                profiles = self.catalog.get_profiles()
                if profiles:
                    gr.Markdown("**Доступные профили:**")
                    for name, data in profiles.items():
                        desc = data.get("description", "")
                        tags = data.get("tags", [])
                        gr.Markdown(f"- **{name}**: {desc} (tags: {', '.join(tags)})")

                gr.Markdown("---")

                hf_token_set = "установлен" if os.environ.get("HF_TOKEN") else "НЕ установлен"
                gr.Textbox(
                    label="HuggingFace Token",
                    value=f"Статус: {hf_token_set}",
                    interactive=False,
                )
                gr.Markdown(
                    "Токен задаётся через переменную окружения `HF_TOKEN` в RunPod Template."
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
