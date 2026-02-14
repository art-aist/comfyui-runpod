"""
Model Downloader — downloads models from HuggingFace with progress reporting.
Used by the Manager UI for real-time progress display.
"""

import os
import time
import threading
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import hf_hub_download, login


class DownloadProgress:
    """Track download progress for UI updates."""

    def __init__(self):
        self.current_model: str = ""
        self.current_index: int = 0
        self.total_models: int = 0
        self.downloaded_bytes: int = 0
        self.total_bytes: int = 0
        self.speed_mbps: float = 0
        self.status: str = "idle"  # idle, downloading, done, error
        self.log_lines: list = []
        self._lock = threading.Lock()

    def log(self, message: str):
        with self._lock:
            self.log_lines.append(message)
            # Keep last 100 lines
            if len(self.log_lines) > 100:
                self.log_lines = self.log_lines[-100:]

    def get_log(self) -> str:
        with self._lock:
            return "\n".join(self.log_lines)

    def get_summary(self) -> str:
        if self.status == "idle":
            return "Ожидание..."
        elif self.status == "downloading":
            return f"Скачивание: {self.current_model} ({self.current_index}/{self.total_models})"
        elif self.status == "done":
            return f"Завершено: {self.total_models} моделей"
        elif self.status == "error":
            return "Ошибка при скачивании"
        return self.status


class ModelDownloader:
    """Downloads models from HuggingFace."""

    def __init__(self, comfyui_path: str):
        self.comfyui_path = Path(comfyui_path)
        self.progress = DownloadProgress()
        self._download_thread: Optional[threading.Thread] = None

    def is_downloading(self) -> bool:
        return self._download_thread is not None and self._download_thread.is_alive()

    def download_models(self, models: list, hf_token: str = None):
        """Start downloading models in a background thread."""
        if self.is_downloading():
            self.progress.log("Скачивание уже идёт, подождите...")
            return

        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(models, hf_token),
            daemon=True,
        )
        self._download_thread.start()

    def _download_worker(self, models: list, hf_token: str = None):
        """Background worker for downloading models."""
        self.progress.status = "downloading"
        self.progress.total_models = len(models)
        self.progress.log_lines = []

        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=False)
                self.progress.log("HuggingFace: authenticated")
            except Exception as e:
                self.progress.log(f"HF auth warning: {e}")

        downloaded = 0
        skipped = 0
        failed = 0

        for i, model in enumerate(models):
            self.progress.current_index = i + 1
            self.progress.current_model = model["name"]

            dest_dir = self.comfyui_path / model["dest_path"]
            local_path = dest_dir / model["filename"]
            size_gb = model.get("size_gb", 0)

            # Skip existing
            if local_path.exists():
                file_size_gb = local_path.stat().st_size / (1024 ** 3)
                if size_gb == 0 or abs(file_size_gb - size_gb) < 0.5:
                    self.progress.log(f"[{i+1}/{len(models)}] {model['name']} — уже есть")
                    skipped += 1
                    continue

            self.progress.log(
                f"[{i+1}/{len(models)}] {model['name']} ({size_gb}GB)..."
            )

            dest_dir.mkdir(parents=True, exist_ok=True)
            start = time.time()

            try:
                source = model.get("source", "hf")
                if source in ("hf", "huggingface"):
                    hf_hub_download(
                        repo_id=model["hf_repo"],
                        filename=model.get("hf_file", model["filename"]),
                        local_dir=str(dest_dir),
                        local_dir_use_symlinks=False,
                        token=hf_token,
                    )
                elif source == "url":
                    import urllib.request
                    urllib.request.urlretrieve(model["url"], str(local_path))

                elapsed = time.time() - start
                speed = size_gb / elapsed * 1024 if elapsed > 0 else 0
                self.progress.log(
                    f"  OK ({elapsed:.0f}s, {speed:.0f} MB/s)"
                )
                downloaded += 1

            except Exception as e:
                self.progress.log(f"  ОШИБКА: {str(e)[:200]}")
                failed += 1

        self.progress.log("")
        self.progress.log(f"Итого: скачано {downloaded}, были {skipped}, ошибки {failed}")
        self.progress.status = "done" if failed == 0 else "error"
