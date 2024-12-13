#!/bin/bash

DIRECTORY="results/random_walks_llm"

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Iterate over all CSV files in the directory
for file in "$DIRECTORY"/*.csv; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        python irt_analysis.py --random_walk_file "$file"
    else
        echo "No CSV files found in $DIRECTORY"
    fi
done

echo "Finished calculating IRT"
