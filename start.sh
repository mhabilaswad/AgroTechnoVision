#!/bin/bash

echo "===================================="
echo "  Fruit Detection System - Startup"
echo "===================================="
echo ""

echo "[1/3] Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found! Please install Python 3.8+"
    exit 1
fi
echo ""

echo "[2/3] Installing dependencies..."
pip3 install -r requirements.txt
echo ""

echo "[3/3] Starting server..."
echo ""
echo "===================================="
echo "  Server is running!"
echo "  Open your browser at:"
echo "  http://localhost:5000"
echo "===================================="
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
