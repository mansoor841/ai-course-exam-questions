from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from typing import Optional
import uuid

from pdf_llm_trainer import (
    PDFToLLMTrainer,
    BasicPDFProcessor,
    DocumentEmbedder,
    DocumentRepository
)

app = FastAPI(
    title="PDF-to-LLM Service",
    description="API for training LLMs on PDF documents and generating responses",
    version="0.1.0"
)

TRAINED_MODELS_DIR = "./trained_models"
DOCUMENT_CACHE_DIR = "./document_cache"
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(DOCUMENT_CACHE_DIR, exist_ok=True)

model_registry = {}

class TrainRequest(BaseModel):
    model_name: Optional[str] = "gpt2"
    chunk_size: Optional[int] = 800
    overlap: Optional[int] = 100
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 4
    learning_rate: Optional[float] = 5e-5

class TrainResponse(BaseModel):
    model_id: str
    status: str
    message: str

class GenerateRequest(BaseModel):
    model_id: str
    prompt: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.7

class GenerateResponse(BaseModel):
    generated_text: str
    model_id: str

@app.post("/train/", response_model=TrainResponse)
async def train_model(
    file: UploadFile = File(...),
    config: TrainRequest = TrainRequest()
):
    try:
        model_id = str(uuid.uuid4())
        model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
        
        file_path = f"/tmp/{model_id}.pdf"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        pdf_processor = BasicPDFProcessor(
            chunk_size=config.chunk_size,
            overlap=config.overlap
        )
        embedder = DocumentEmbedder()
        doc_repo = DocumentRepository(cache_dir=DOCUMENT_CACHE_DIR)
        
        trainer = PDFToLLMTrainer(
            base_model_name=config.model_name,
            pdf_processor=pdf_processor,
            embedder=embedder
        )
        
        chunks = doc_repo.load_document_chunks(file_path)
        
        if not chunks:
            chunks = trainer.process_pdf(file_path)
            doc_repo.save_document_chunks(file_path, chunks)
        
        trainer.train(
            chunks=chunks,
            output_dir=model_dir,
            num_epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate
        )
        
        model_registry[model_id] = {
            "model_path": model_dir,
            "trainer": trainer
        }
        
        os.remove(file_path)
        
        return TrainResponse(
            model_id=model_id,
            status="success",
            message=f"Model trained successfully and saved as {model_id}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@app.post("/generate/", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        if request.model_id not in model_registry:
            model_path = os.path.join(TRAINED_MODELS_DIR, request.model_id)
            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {request.model_id} not found"
                )
            
            trainer = PDFToLLMTrainer.load_model(model_path)
            model_registry[request.model_id] = {
                "model_path": model_path,
                "trainer": trainer
            }
        
        trainer = model_registry[request.model_id]["trainer"]
        
        generated_text = trainer.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return GenerateResponse(
            generated_text=generated_text,
            model_id=request.model_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

@app.get("/models/")
async def list_models():
    registered_models = list(model_registry.keys())
    disk_models = [
        d for d in os.listdir(TRAINED_MODELS_DIR)
        if os.path.isdir(os.path.join(TRAINED_MODELS_DIR, d))
    ]
    
    all_models = list(set(registered_models + disk_models))
    
    return JSONResponse(
        content={
            "models": all_models,
            "count": len(all_models)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)