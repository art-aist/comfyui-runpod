#!/bin/bash
# =============================================================
# Snapshot — сохраняет состояние перед установкой nodes
# Usage: snapshot.sh create <name> <comfyui_path>
#        snapshot.sh restore <name> <comfyui_path>
#        snapshot.sh list
# =============================================================

ACTION="${1:?Usage: snapshot.sh <create|restore|list> [name] [comfyui_path]}"
NAME="${2:-}"
COMFYUI_PATH="${3:-/workspace/ComfyUI}"
SNAPSHOT_DIR="/workspace/.snapshots"

mkdir -p "$SNAPSHOT_DIR"

case "$ACTION" in
    create)
        [ -z "$NAME" ] && NAME="snapshot_$(date +%Y%m%d_%H%M%S)"
        DEST="$SNAPSHOT_DIR/$NAME"
        mkdir -p "$DEST"

        echo "Creating snapshot: $NAME"

        # Save custom_nodes list
        ls -la "$COMFYUI_PATH/custom_nodes/" > "$DEST/nodes_list.txt" 2>/dev/null
        for d in "$COMFYUI_PATH/custom_nodes"/*/; do
            if [ -d "$d/.git" ]; then
                basename=$(basename "$d")
                commit=$(cd "$d" && git rev-parse --short HEAD 2>/dev/null)
                echo "$basename $commit" >> "$DEST/nodes_commits.txt"
            fi
        done

        # Save pip freeze
        pip freeze > "$DEST/pip_freeze.txt" 2>/dev/null

        # Save timestamp
        date > "$DEST/timestamp.txt"

        echo "Snapshot saved: $DEST"
        echo "Files:"
        ls -la "$DEST/"
        ;;

    restore)
        [ -z "$NAME" ] && echo "Error: specify snapshot name" && exit 1
        SRC="$SNAPSHOT_DIR/$NAME"
        [ ! -d "$SRC" ] && echo "Error: snapshot '$NAME' not found" && exit 1

        echo "Restoring snapshot: $NAME"
        echo "WARNING: this will reinstall pip packages to match the snapshot"
        echo "Press Ctrl+C to cancel, Enter to continue..."
        read

        # Restore pip packages
        if [ -f "$SRC/pip_freeze.txt" ]; then
            pip install -r "$SRC/pip_freeze.txt" 2>/dev/null
        fi

        echo "Snapshot restored. You may need to restart ComfyUI."
        ;;

    list)
        echo "Available snapshots:"
        for d in "$SNAPSHOT_DIR"/*/; do
            [ -d "$d" ] || continue
            name=$(basename "$d")
            ts=$(cat "$d/timestamp.txt" 2>/dev/null || echo "unknown")
            echo "  $name — $ts"
        done
        ;;

    *)
        echo "Unknown action: $ACTION"
        echo "Usage: snapshot.sh <create|restore|list> [name] [comfyui_path]"
        exit 1
        ;;
esac
