#!/usr/bin/env bash

ROOT_DIR="$1"

if [ -z "$ROOT_DIR" ]; then
  echo "Usage: $0 <root_dir>"
  exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
  echo "Error: $ROOT_DIR is not a directory"
  exit 1
fi

for dir in "$ROOT_DIR"/*/; do
  [ -d "$dir" ] || continue
  last_dirs=$(find "$dir" -type d -links 2)
  for last in $last_dirs; do
    file_count=$(find "$last" -maxdepth 1 -type f | wc -l)
    echo "$last : $file_count"
  done
done

