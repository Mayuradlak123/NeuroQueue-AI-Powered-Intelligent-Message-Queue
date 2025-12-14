#!/bin/bash

echo "🚀 Starting NeuroQueue..."

# Try to activate venv
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run Entry Point
# We use main.py because it initializes AI models before starting the server
python main.py