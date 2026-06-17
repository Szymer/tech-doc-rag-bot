# control-cabinet-rag

Minimalny szkielet projektu RAG dla dokumentacji technicznej szaf sterowniczych.

## Start z uv

```bash
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpointy

- `GET /health`
- `POST /api/ask`

## Devcontainer + debug

Repo zawiera konfigurację `.devcontainer/` oraz `.vscode/launch.json` pod debug FastAPI.
