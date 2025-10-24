#!/bin/bash
# Quick Start Script for Sentiment Analysis Project

set -e

echo "======================================"
echo "Sentiment Analysis - Quick Start"
echo "======================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

echo "Step 1: Verifying setup..."
python setup_check.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Setup verification failed. Please install dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "Step 2: Choose deployment method:"
echo "  1) Local (requires trained model)"
echo "  2) Docker (recommended)"
echo "  3) Train model first"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting local API server..."
        echo "API will be available at http://localhost:8000"
        echo "Documentation at http://localhost:8000/docs"
        echo ""
        python app.py
        ;;
    2)
        if ! command -v docker &> /dev/null; then
            echo "Error: Docker is not installed"
            exit 1
        fi
        echo ""
        echo "Building and starting Docker container..."
        docker-compose up --build -d
        echo ""
        echo "[PASS] Container started successfully!"
        echo "API: http://localhost:8000"
        echo "Docs: http://localhost:8000/docs"
        echo ""
        echo "View logs: docker-compose logs -f"
        echo "Stop: docker-compose down"
        ;;
    3)
        echo ""
        echo "Starting model training..."
        echo "This may take 30-60 minutes on GPU, 2-3 hours on CPU"
        echo ""
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            python train.py
            echo ""
            echo "[PASS] Training completed!"
            echo "Model saved to: models/final_model/"
            echo ""
            echo "Now you can start the API server:"
            echo "  python app.py"
            echo "or"
            echo "  docker-compose up -d"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
