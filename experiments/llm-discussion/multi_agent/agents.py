
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import threading
from typing import List, Dict, Any

import torch
from openai import OpenAI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Global Hugging Face Mistral model objects (shared across all MistralAgent instances)
_mistral_model = None
_mistral_tokenizer = None
_mistral_generator = None
_loaded_model_id = None
# Lock to ensure only one generation happens at a time (prevents memory spikes from concurrent generations)
# Using RLock (reentrant) to allow recursive calls in error handling
_mistral_generation_lock = threading.RLock()

def _load_mistral_model(model_id="mistralai/Mistral-7B-Instruct-v0.2"):
    """Load Mistral model globally (lazy loading on first use, shared across all agents)"""
    global _mistral_model, _mistral_tokenizer, _mistral_generator, _loaded_model_id
    
    if _mistral_model is None:
        print(f"[MistralAgent] Loading Mistral model globally: {model_id}")
        print("[MistralAgent] This model will be shared across all MistralAgent instances")
        HF_TOKEN = os.getenv("HUGGINGFACE_API_key") or os.getenv("HUGGINGFACE_API_KEY")
        
        _mistral_tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=HF_TOKEN
        )
        _mistral_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            token=HF_TOKEN
        )
        _mistral_generator = pipeline(
            "text-generation",
            model=_mistral_model,
            tokenizer=_mistral_tokenizer,
            device_map="auto",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16}
        )
        _loaded_model_id = model_id
        print(f"[MistralAgent] Model loaded successfully: {model_id}")
    else:
        # Model already loaded, verify it's the same model_id
        if model_id != _loaded_model_id:
            print(f"[MistralAgent] Warning: Requested model_id '{model_id}' differs from already loaded model '{_loaded_model_id}'. Using existing model.")
    
    return _mistral_tokenizer, _mistral_model, _mistral_generator

# Ensure the HF hub knows about your token
def generate_response_llama2_torchrun(
    message,
    ckpt_dir: str = "/tmp2/llama-2-7b-chat",
    tokenizer_path: str = "/home/chenlawrance/repo/LLM-Creativity/model/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4):
    message_json = json.dumps(message)  # Serialize the message to a JSON string
    command = [
        "torchrun", "--nproc_per_node=1", "/home/chenlawrance/repo/LLM-Creativity/llama_model/llama_chat_completion.py",
        "--ckpt_dir", ckpt_dir,
        "--tokenizer_path", tokenizer_path,
        "--max_seq_len", str(max_seq_len),
        "--max_batch_size", str(max_batch_size),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--message", message_json
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()

        # Find the beginning of the generated response
        assistant_prefix = "> Assistant:"
        start_idx = output.find(assistant_prefix)
        if start_idx != -1:
            # Calculate the starting index of the actual response
            start_of_response = start_idx + len(assistant_prefix)
            # Extract and return the generated response part
            generated_response = output[start_of_response:].strip()
            return generated_response
        else:
            return "No response generated or unable to extract response."
    except subprocess.CalledProcessError as e:
        print(f"Error executing torchrun command: {e.stderr}")
        return "Unable to generate response due to an error."

class Agent:
    def generate_answer(self, answer_context):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_assistant_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_user_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")

class OpenAIAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate, missing_history = []):
        self.model_name = model_name
        self.client = OpenAI()
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history
        
    def generate_answer(self, answer_context, temperature=1):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=answer_context,
                n=1)
            result = completion.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(10)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}
    
class GeminiAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai is not installed. Install it with: pip install google-generativeai")
        self.model_name = model_name
        genai.configure(api_key=os.environ["GEMINI_API_KEY"]) # ~/.bashrc save : export GEMINI_API_KEY="YOUR_API" 
        self.model = genai.GenerativeModel(self.model_name)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate

    def generate_answer(self, answer_context,temperature= 1.0):
        try: 
            response = self.model.generate_content(
                answer_context,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT","threshold": "BLOCK_NONE",},
                    {"category": "HARM_CATEGORY_HATE_SPEECH","threshold": "BLOCK_NONE",},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_NONE",},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_NONE",},
                    ]
            )
            # for pure text -> return response.text
            # return response.candidates[0].content
            return response.text
        except Exception as e:
            logging.exception("Exception occurred during response generation: " + str(e))
            time.sleep(1)
            return self.generate_answer(answer_context)
        
    def construct_assistant_message(self, content):
        response = {"role": "model", "parts": [content]}
        return response
    
    def construct_user_message(self, content):
        response = {"role": "user", "parts": [content]}
        return response
        
class Llama2Agent(Agent):
    def __init__(self, ckpt_dir, tokenizer_path, agent_name):
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.agent_name = agent_name

    def generate_answer(self, answer_context, temperature=0.6, top_p=0.9, max_seq_len=100000, max_batch_size=4): # return pure text
        return generate_response_llama2_torchrun(
            message=answer_context,
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            temperature=temperature,
            top_p=top_p,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size
        )
    
    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}
    
class OpenAIAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate, missing_history = []):
        self.model_name = model_name
        self.client = OpenAI()
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history
        
    def generate_answer(self, answer_context, temperature=1):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=answer_context,
                n=1)
            result = completion.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(10)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}

class MistralAgent(Agent):
    """
    MistralAgent uses a globally shared Hugging Face model instance.
    All MistralAgent instances share the same model, tokenizer, and generator
    to avoid loading the model multiple times in memory.
    """
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate, missing_history = [], model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.model_id = model_id
        # Load or reuse the global model (shared across all MistralAgent instances)
        self.tokenizer, self.model, self.generator = _load_mistral_model(model_id)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history
        
    def generate_answer(self, answer_context, temperature=1):
        # Use lock to ensure only one generation at a time (prevents memory spikes)
        with _mistral_generation_lock:
            try:
                # Convert chat messages to Mistral format using tokenizer's chat template
                # answer_context is a list of dicts with "role" and "content" keys
                formatted_prompt = self.tokenizer.apply_chat_template(
                    answer_context,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Generate response using the pipeline
                outputs = self.generator(
                    formatted_prompt,
                    max_new_tokens=256,  # Reduced from 512 to save memory
                    temperature=temperature,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract the generated text
                result = outputs[0]["generated_text"].strip()
                
                # Clear cache to free memory (especially important for MPS)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return result
            except Exception as e:
                print(f"Error with model {self.model_name}: {e}")
                # Clear cache on error too
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(10)
                return self.generate_answer(answer_context, temperature)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}

__all__ = [
    "Agent",
    "OpenAIAgent",
    "GeminiAgent",
    "Llama2Agent",
    "LocalMistral7bAgent",
    "MistralAgent"
]
