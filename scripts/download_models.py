#!/usr/bin/env python3
"""
Download models from HuggingFace based on catalog and profile.

Usage:
    python3 download_models.py --catalog config/models_catalog.json --profile wan_video --comfyui-path /workspace/ComfyUI
    python3 download_models.py --catalog config/models_catalog.json --profile wan_video --dry-run

Environment:
    HF_TOKEN — HuggingFace token for private/gated repos
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def load_catalog(catalog_path: str) -> dict:
    with open(catalog_path) as f:
        return json.load(f)


def get_models_for_profile(catalog: dict, profile_name: str, tier_filter: list = None) -> list:
    """Get list of models matching profile tags and optional tier filter."""
    profile = catalog.get("profiles", {}).get(profile_name, {})
    profile_tags = set(profile.get("tags", []))

    models = []
    for cat_name, cat in catalog.get("categories", {}).items():
        for model in cat.get("models", []):
            # Check tier filter
            if tier_filter and model.get("tier") not in tier_filter:
                continue

            # Check tag match
            model_tags = set(model.get("tags", []))
            if "*" in profile_tags or model_tags & profile_tags:
                model_copy = dict(model)
                model_copy["_category"] = cat_name
                models.append(model_copy)

    return models


def download_model_hf(model: dict, base_path: str, hf_token: str = None) -> dict:
    """Download a single model from HuggingFace."""
    from huggingface_hub import hf_hub_download

    dest_dir = Path(base_path) / model["dest_path"]
    filename = model["filename"]
    local_path = dest_dir / filename

    result = {
        "name": model["name"],
        "filename": filename,
        "size_gb": model.get("size_gb", 0),
        "status": "unknown",
        "time_seconds": 0,
    }

    # Skip if already exists
    if local_path.exists():
        file_size_gb = local_path.stat().st_size / (1024 ** 3)
        expected = model.get("size_gb", 0)
        if expected == 0 or abs(file_size_gb - expected) < 0.5:
            result["status"] = "exists"
            return result

    # Create directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    hf_repo = model.get("hf_repo", "")
    hf_file = model.get("hf_file", "")

    if not hf_repo or not hf_file:
        result["status"] = "error:no_hf_repo_or_file"
        return result

    start = time.time()

    try:
        downloaded_path = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_file,
            local_dir=str(dest_dir),
            token=hf_token,
        )

        # Move file to expected location if hf_file has subdirectories
        downloaded = Path(downloaded_path)
        if downloaded != local_path and downloaded.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(local_path)

        elapsed = time.time() - start
        result["status"] = "downloaded"
        result["time_seconds"] = round(elapsed, 1)

    except Exception as e:
        result["status"] = f"error:{str(e)[:200]}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace")
    parser.add_argument("--catalog", required=True, help="Path to models_catalog.json")
    parser.add_argument("--profile", default="default", help="Profile name (wan_video, flux, full)")
    parser.add_argument("--comfyui-path", default="/workspace/ComfyUI", help="ComfyUI base path")
    parser.add_argument("--tier", type=int, nargs="+", help="Filter by tier (e.g. --tier 2 for auto-download)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    # Load catalog
    catalog = load_catalog(args.catalog)
    models = get_models_for_profile(catalog, args.profile, args.tier)

    if not models:
        print(f"No models found for profile '{args.profile}'" +
              (f" tier={args.tier}" if args.tier else ""))
        sys.exit(0)

    # Summary
    total_gb = sum(m.get("size_gb", 0) for m in models)
    print(f"\nProfile: {args.profile}")
    print(f"Models:  {len(models)}")
    print(f"Total:   {total_gb:.1f}GB")
    print(f"Dest:    {args.comfyui_path}")
    if args.dry_run:
        print(f"Mode:    DRY RUN")
    print("")

    for i, model in enumerate(models, 1):
        source = model.get("source", "hf")
        tier = model.get("tier", "?")
        size = model.get("size_gb", 0)
        dest = Path(args.comfyui_path) / model["dest_path"] / model["filename"]
        exists = dest.exists()

        status = "EXISTS" if exists else "DOWNLOAD"
        print(f"  [{i}/{len(models)}] T{tier} {model['name']} ({size}GB) [{source}] -> {status}")

    if args.dry_run:
        print(f"\nDry run complete. Use without --dry-run to download.")
        sys.exit(0)

    # HuggingFace auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("\nHuggingFace: authenticated")
        except Exception:
            print("\nHuggingFace: token set but login failed, continuing...")
    else:
        print("\nHuggingFace: anonymous (set HF_TOKEN for private/gated repos)")

    # Download
    downloaded = 0
    skipped = 0
    failed = 0

    for i, model in enumerate(models, 1):
        source = model.get("source", "hf")
        print(f"\n[{i}/{len(models)}] {model['name']} ({model.get('size_gb', 0)}GB)...", end=" ", flush=True)

        if source not in ("hf",):
            print(f"SKIP (source={source}, not supported yet)")
            skipped += 1
            continue

        result = download_model_hf(model, args.comfyui_path, hf_token)

        if result["status"] == "exists":
            print("already exists")
            skipped += 1
        elif result["status"] == "downloaded":
            print(f"OK ({result['time_seconds']}s)")
            downloaded += 1
        else:
            print(f"ERROR: {result['status']}")
            failed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"Downloaded: {downloaded}, Existed: {skipped}, Errors: {failed}")
    print(f"Total: {downloaded + skipped}/{len(models)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
