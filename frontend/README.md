# Go Frontend for Sentiment Analysis

Simple Go web interface for testing the sentiment analysis API.

## Run

```bash
cd frontend
go run main.go
```

Then open: http://localhost:3000

## Requirements

- Go 1.16+
- FastAPI backend running on port 8000
- Training results in ../results/ directory

## Features

- Single review prediction with confidence scores
- Batch prediction (multiple reviews)
- Display training visualizations (EDA, metrics)
- Real-time API health status
- Uses Macromill styling

## Ports

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
