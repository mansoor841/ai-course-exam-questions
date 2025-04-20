from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Generator
import warnings

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

class DocumentError(Exception):
    pass

class ModelTrainingError(Exception):
    pass

@dataclass
class DocumentChunk:
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    page_number: Optional[int] = None

class PDFProcessor(ABC):
    @abstractmethod
    def process(self, file_path: str) -> List[DocumentChunk]:
        pass

class BasicPDFProcessor(PDFProcessor):
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def process(self, file_path: str) -> List[DocumentChunk]:
        if not os.path.exists(file_path):
            raise DocumentError(f"File not found: {file_path}")
            
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        continue
                        
                    page_chunks = self._chunk_text(text)
                    chunks.extend([
                        DocumentChunk(
                            text=chunk,
                            metadata={
                                'source': file_path,
                                'page': page_num
                            },
                            page_number=page_num
                        )
                        for chunk in page_chunks
                    ])
                    
        except Exception as e:
            raise DocumentError(f"Failed to process PDF: {str(e)}")
            
        return chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap
            
        return chunks

class DocumentEmbedder:    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        
    def embed_documents(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        return chunks

class DocumentDataset(Dataset):
    def __init__(self, chunks: List[DocumentChunk], tokenizer: AutoTokenizer, max_length: int = 512):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        encoding = self.tokenizer(
            chunk.text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class PDFToLLMTrainer:
    def __init__(
        self,
        base_model_name: str = 'gpt2',
        pdf_processor: Optional[PDFProcessor] = None,
        embedder: Optional[DocumentEmbedder] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.base_model_name = base_model_name
        self.device = device
        
        self.pdf_processor = pdf_processor or BasicPDFProcessor()
        self.embedder = embedder or DocumentEmbedder(device=device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        
        self._is_trained = False
        self._training_metrics = {}
        
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        chunks = self.pdf_processor.process(file_path)
        return self.embedder.embed_documents(chunks)
    
    def train(
        self,
        chunks: List[DocumentChunk],
        output_dir: str = './results',
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        test_size: float = 0.1,
        save_steps: int = 500,
        logging_steps: int = 100
    ) -> None:
        if not chunks:
            raise ModelTrainingError("No document chunks provided for training")
            
        train_chunks, val_chunks = train_test_split(chunks, test_size=test_size)
        
        train_dataset = DocumentDataset(train_chunks, self.tokenizer)
        val_dataset = DocumentDataset(val_chunks, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=True,
            save_total_limit=2,
            report_to="none",
            disable_tqdm=False
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        try:
            print("Starting training...")
            trainer.train()
            
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            self._is_trained = True
            self._training_metrics = trainer.state.log_history
            
        except Exception as e:
            raise ModelTrainingError(f"Training failed: {str(e)}")
    
    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        if not self._is_trained:
            warnings.warn("Model has not been trained yet. Using base model behavior.")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save_model(self, output_dir: str) -> None:
        if not self._is_trained:
            raise ModelTrainingError("Model has not been trained yet")
            
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def load_model(cls, model_dir: str, device: str = 'auto') -> 'PDFToLLMTrainer':
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        
        instance = cls(base_model_name=model_dir, device=device)
        instance.tokenizer = tokenizer
        instance.model = model
        instance._is_trained = True
        
        return instance
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    @property
    def training_metrics(self) -> Dict:
        return self._training_metrics

class DocumentRepository:
    def __init__(self, cache_dir: str = './document_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, file_path: str) -> Path:
        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        return self.cache_dir / f"{file_hash}.pkl"
    
    def save_document_chunks(self, file_path: str, chunks: List[DocumentChunk]) -> None:
        cache_path = self._get_cache_path(file_path)
        pd.to_pickle(chunks, cache_path)
    
    def load_document_chunks(self, file_path: str) -> Optional[List[DocumentChunk]]:
        cache_path = self._get_cache_path(file_path)
        if cache_path.exists():
            return pd.read_pickle(cache_path)
        return None
