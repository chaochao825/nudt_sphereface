#!/bin/bash
# wind_metrics.sh - Read training stats and output as JSON
# This script is called by external processes to get training status

# Default stats file location
STATS_FILE="${OUTPUT_PATH:-/project/output}/stats.json"

# Check if stats file exists
if [ ! -f "$STATS_FILE" ]; then
    # Return empty/default stats if file doesn't exist
    echo '{
  "status": "not_started",
  "message": "Training has not started or stats file not found"
}'
    exit 0
fi

# Read and output the stats.json file
cat "$STATS_FILE"

