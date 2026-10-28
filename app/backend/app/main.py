from fastapi import FastAPI

app = FastAPI(title="Ether API")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
