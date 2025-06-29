"""
Model manager for automatic download and management of language models
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import warnings

try:
    from ..config.settings import MODEL_CONFIG
except ImportError:
    try:
        from config.settings import MODEL_CONFIG
    except ImportError:
        MODEL_CONFIG = {
            "name": "teknium/OpenHermes-2.5-Mistral-7B",
            "device_map": "auto",
            "torch_dtype": "float16",
            "load_in_4bit": True,
            "max_new_tokens": 512,  # Updated to use max_new_tokens
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": 2,
            "cache_dir": "./models",
            "use_auth_token": None,
            "trust_remote_code": True,
            "generation_config": {
                "max_new_tokens": 512,
                "min_new_tokens": 10,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0
            }
        }

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelManager:
    """Manages model loading, downloading, and inference"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or MODEL_CONFIG.copy()
        self.logger = logging.getLogger("ModelManager")
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Device configuration
        self.device = self._setup_device()
        
        # Model state
        self.is_loaded = False
        self.model_name = self.config["name"]
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            "name": "teknium/OpenHermes-2.5-Mistral-7B",
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "max_new_tokens": 512,  # Use max_new_tokens instead of max_length
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": 2,  # EOS token for Mistral
            "cache_dir": "./models",
            "use_auth_token": False,
        }
    
    def _setup_device(self) -> str:
        """Setup computing device"""
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            self.logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            self.logger.info("Using CPU (this will be slow for large models)")
        
        return device
    
    def check_model_availability(self) -> Dict[str, bool]:
        """Check if models are available locally or need downloading"""
        cache_dir = Path(self.config["cache_dir"])
        model_path = cache_dir / self.model_name.replace("/", "--")
        
        return {
            "model_cached": model_path.exists(),
            "huggingface_available": True,  # Assume HF is available
            "local_storage_gb": self._get_available_storage_gb(),
            "estimated_model_size_gb": 13.0,  # Mistral 7B in 4-bit is ~13GB
        }
    
    def _get_available_storage_gb(self) -> float:
        """Get available storage space in GB"""
        try:
            import shutil
            _, _, free = shutil.disk_usage(".")
            return free / (1024**3)  # Convert to GB
        except Exception:
            return 100.0  # Assume sufficient space if check fails
    
    def load_model(self, force_reload: bool = False) -> bool:
        """Load the model and tokenizer"""
        if self.is_loaded and not force_reload:
            return True
        
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Check availability
            availability = self.check_model_availability()
            if not availability["model_cached"]:
                self.logger.info("Model not cached locally, will download from Hugging Face")
                if availability["local_storage_gb"] < availability["estimated_model_size_gb"]:
                    self.logger.warning(
                        f"Low disk space: {availability['local_storage_gb']:.1f}GB available, "
                        f"need ~{availability['estimated_model_size_gb']}GB"
                    )
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.config.get("load_in_4bit", False) and self.device == "cuda":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    self.logger.info("Using 4-bit quantization for memory efficiency")
                except Exception as e:
                    self.logger.warning(f"Failed to configure quantization: {e}. Loading without quantization.")
                    quantization_config = None
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.config["cache_dir"],
                use_auth_token=self.config["use_auth_token"]
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.logger.info("Loading model (this may take a few minutes)...")
            
            # Handle torch_dtype conversion
            torch_dtype = self.config.get("torch_dtype", torch.float16)
            if isinstance(torch_dtype, str):
                dtype_map = {
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16
                }
                torch_dtype = dtype_map.get(torch_dtype, torch.float16)
            
            model_kwargs = {
                "cache_dir": self.config["cache_dir"],
                "torch_dtype": torch_dtype,
                "use_auth_token": self.config.get("use_auth_token"),
                "low_cpu_mem_usage": True,
                "trust_remote_code": self.config.get("trust_remote_code", True),
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = self.config.get("device_map", "auto")
            else:
                # For non-quantized models, load to specific device
                model_kwargs["device_map"] = None
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except Exception as e:
                self.logger.error(f"Failed to load model with current config: {e}")
                # Try loading without quantization as fallback
                self.logger.info("Attempting to load without quantization...")
                model_kwargs_fallback = {
                    "cache_dir": self.config["cache_dir"],
                    "torch_dtype": torch.float16,
                    "use_auth_token": self.config["use_auth_token"],
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    "device_map": None
                }
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs_fallback
                    )
                    self.logger.info("Successfully loaded model without quantization")
                    quantization_config = None  # Update flag for device placement
                except Exception as e2:
                    self.logger.error(f"Failed to load model even without quantization: {e2}")
                    raise e2
            
            # Move to device if not using device_map
            if not quantization_config:
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier text generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.config["device_map"] if quantization_config else None,
                torch_dtype=self.config["torch_dtype"],
            )
            
            self.is_loaded = True
            self.logger.info("Model loaded successfully!")
            
            # Log model info
            if hasattr(self.model, 'num_parameters'):
                param_count = self.model.num_parameters() / 1e9
                self.logger.info(f"Model parameters: {param_count:.1f}B")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate response using the loaded model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not specified
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),  # Use max_new_tokens instead of max_length
            "min_new_tokens": kwargs.get("min_new_tokens", 10),
            "temperature": temperature or self.config["temperature"],
            "top_p": top_p or self.config["top_p"],
            "do_sample": self.config["do_sample"],
            "pad_token_id": self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else self.config["pad_token_id"],
            "eos_token_id": self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else None,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "length_penalty": kwargs.get("length_penalty", 1.0),
            "num_return_sequences": 1,
            "return_full_text": False,
            **kwargs
        }
        
        try:
            # Generate response
            with torch.no_grad():
                outputs = self.pipeline(
                    prompt,
                    **generation_config
                )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Clean up the response
            response = self._clean_response(generated_text, prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def _clean_response(self, generated_text: str, original_prompt: str) -> str:
        """Clean and format the generated response"""
        # Remove the original prompt if it's repeated
        if generated_text.startswith(original_prompt):
            response = generated_text[len(original_prompt):].strip()
        else:
            response = generated_text.strip()
        
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        
        # Limit response length
        sentences = response.split('. ')
        if len(sentences) > 3:  # Limit to 3 sentences for conversation
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    def format_conversation_prompt(
        self,
        agent_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        current_message: str
    ) -> str:
        """Format a conversation prompt for the model"""
        
        # Build system prompt based on agent context
        system_parts = []
        system_parts.append(f"You are {agent_context['agent_name']}, a {agent_context['agent_role']}.")
        system_parts.append(f"Your personality: {agent_context['personality_description']}")
        
        if agent_context.get('dominant_emotion') and agent_context.get('emotion_intensity', 0) > 0.3:
            emotion = agent_context['dominant_emotion']
            intensity = agent_context['emotion_intensity']
            system_parts.append(f"You are currently feeling {emotion} (intensity: {intensity:.1f}).")
        
        if agent_context.get('current_goal'):
            system_parts.append(f"Your current goal: {agent_context['current_goal']}")
        
        if agent_context.get('relationship_context'):
            system_parts.append(f"Relationship context: {agent_context['relationship_context']}")
        
        if agent_context.get('relevant_memories'):
            memories = agent_context['relevant_memories'][:2]  # Limit memories
            if memories:
                system_parts.append(f"Relevant memories: {'; '.join(memories)}")
        
        system_prompt = " ".join(system_parts)
        
        # Build conversation history
        history_parts = []
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            speaker = msg.get('speaker', 'Unknown')
            content = msg.get('content', '')
            history_parts.append(f"{speaker}: {content}")
        
        # Build the full prompt
        if history_parts:
            conversation_context = "\n".join(history_parts)
            prompt = f"<s>[INST] {system_prompt}\n\nConversation so far:\n{conversation_context}\n\nNew message: {current_message}\n\nRespond naturally as {agent_context['agent_name']}: [/INST]"
        else:
            prompt = f"<s>[INST] {system_prompt}\n\nMessage: {current_message}\n\nRespond naturally as {agent_context['agent_name']}: [/INST]"
        
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "config": self.config
        }
        
        if self.is_loaded and self.model:
            try:
                if hasattr(self.model, 'num_parameters'):
                    info["parameters"] = f"{self.model.num_parameters() / 1e9:.1f}B"
                info["model_dtype"] = str(self.model.dtype)
                info["model_device"] = str(self.model.device)
            except Exception:
                pass
        
        return info
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        self.logger.info("Model unloaded successfully")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_loaded:
            self.unload_model()


# Global model manager instance
_model_manager = None

def get_model_manager(config: Optional[Dict[str, Any]] = None) -> ModelManager:
    """Get or create global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(config)
    return _model_manager

def ensure_model_loaded(config: Optional[Dict[str, Any]] = None) -> ModelManager:
    """Ensure model is loaded and return manager"""
    manager = get_model_manager(config)
    if not manager.is_loaded:
        success = manager.load_model()
        if not success:
            raise RuntimeError("Failed to load model")
    return manager
