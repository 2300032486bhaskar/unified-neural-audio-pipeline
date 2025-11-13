from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.post("/submit-offline/")
async def submit_offline(mixture: UploadFile = File(...), target: UploadFile = File(...)):
    return {"status": "FastAPI is working, pipeline disabled temporarily"}

@app.get("/")
def health():
    return {"message": "FastAPI server is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
