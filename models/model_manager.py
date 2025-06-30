import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from typing import Optional, Dict, Any
import os
from huggingface_hub import snapshot_download

class ModelManager:
    """Manages the loading and inference of language models"""
    
    def __init__(self, model_name: str = "teknium/OpenHermes-2.5-Mistral-7B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer with optimization"""
        try:
            logging.info(f"Loading model: {self.model_name}")
            
            # Check if model exists locally, if not download it
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception as e:
                logging.info(f"Downloading model {self.model_name}...")
                snapshot_download(repo_id=self.model_name, local_dir=f"./models/{self.model_name.split('/')[-1]}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Configure for efficient loading
            if self.device == "cuda":
                # Use 4-bit quantization for GPU
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                # CPU loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logging.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 512, 
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """Generate a response using the loaded model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response more carefully
            if prompt in response:
                # Find where the prompt ends and extract only the new generation
                prompt_end = response.find(prompt) + len(prompt)
                response = response[prompt_end:].strip()
            
            # Clean up the response
            response = response.strip()
            
            # If response is empty or too short, provide a fallback
            if not response or len(response.strip()) < 10:
                logging.warning("Generated empty or very short response, using fallback")
                response = "I need more time to process this information."
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A",
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
