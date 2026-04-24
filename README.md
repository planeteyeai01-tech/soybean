# Soybean Detection API

Binary classification API — detects if a field is soybean from a KML file + date.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/predict` | Predict soybean |
| POST | `/retrain` | Retrain model |

## Predict usage

```bash
curl -X POST https://your-app.railway.app/predict \
  -F "file=@field.kml" \
  -F "date=2023-09-12"
```

## Response

```json
{
  "prediction": "Soybean",
  "is_soybean": true,
  "confidence": 0.87,
  "threshold": 0.42,
  "centroid_lat": 19.19,
  "centroid_lon": 77.06,
  "date": "2023-09-12",
  "n_coordinates": 5
}
```

## Deploy on Railway

1. Push this repo to GitHub
2. New project → Deploy from GitHub repo
3. Railway auto-detects Python, installs deps, starts server
4. Model trains automatically on first boot
