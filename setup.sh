#!/bin/bash

echo "🧠 Setting up NeuroQueue Environment..."

# 1. Create venv if not exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv venv
fi

# 2. Activate
# Cross-platform check is hard in bash, assuming standard posix or git bash on windows
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# 3. Pip Install
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Environment Variables
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your keys!"
else
    echo "✅ .env already exists."
fi

echo "✅ Setup Complete!"
echo "🚀 Run './run.sh' to start the system."