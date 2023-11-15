import openai
import gradio as gr

from os import getenv
from typing import Any, Dict, Generator, List

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

temperature = 0.9
top_p = 0.6
repetition_penalty = 1.2

OPENAI_KEY = getenv("OPENAI_API_KEY")
HF_TOKEN = getenv("HUGGING_FACE_HUB_TOKEN")

hf_client = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.1",
        token=HF_TOKEN
        )


def format_prompt(message: str, api_kind: str):
    """
    Formats the given message using a chat template.

    Args:
        message (str): The user message to be formatted.

    Returns:
        str: Formatted message after applying the chat template.
    """

    # Create a list of message dictionaries with role and content
    messages: List[Dict[str, Any]] = [{'role': 'user', 'content': message}]

    if api_kind == "openai":
        return messages
    elif api_kind == "hf":
        return tokenizer.apply_chat_template(messages, tokenize=False)
    elif api_kind:
        raise ValueError("API is not supported")


def generate_hf(prompt: str, history: str, temperature: float = 0.9, max_new_tokens: int = 256,
             top_p: float = 0.95, repetition_penalty: float = 1.0) -> Generator[str, None, str]:
    """
    Generate a sequence of tokens based on a given prompt and history using Mistral client.

    Args:
        prompt (str): The initial prompt for the text generation.
        history (str): Context or history for the text generation.
        temperature (float, optional): The softmax temperature for sampling. Defaults to 0.9.
        max_new_tokens (int, optional): Maximum number of tokens to be generated. Defaults to 256.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
        repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.0.

    Returns:
        Generator[str, None, str]: A generator yielding chunks of generated text.
                                   Returns a final string if an error occurs.
    """

    temperature = max(float(temperature), 1e-2)  # Ensure temperature isn't too low
    top_p = float(top_p)

    generate_kwargs = {
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'do_sample': True,
        'seed': 42,
        }
    
    formatted_prompt = format_prompt(prompt, "hf")

    try:
        stream = hf_client.text_generation(formatted_prompt, **generate_kwargs,
                                            stream=True, details=True, return_full_text=False)
        output = ""
        for response in stream:
            output += response.token.text
            yield output

    except Exception as e:
        if "Too Many Requests" in str(e):
            print("ERROR: Too many requests on Mistral client")
            gr.Warning("Unfortunately Mistral is unable to process")
            return "Unfortunately, I am not able to process your request now."
        elif "Authorization header is invalid" in str(e):
            print("Authetification error:", str(e))
            gr.Warning("Authentication error: HF token was either not provided or incorrect")
            return "Authentication error"
        else:
            print("Unhandled Exception:", str(e))
            gr.Warning("Unfortunately Mistral is unable to process")
            return "I do not know what happened, but I couldn't understand you."


def generate_openai(prompt: str, history: str, temperature: float = 0.9, max_new_tokens: int = 256,
             top_p: float = 0.95, repetition_penalty: float = 1.0) -> Generator[str, None, str]:
    """
    Generate a sequence of tokens based on a given prompt and history using Mistral client.

    Args:
        prompt (str): The initial prompt for the text generation.
        history (str): Context or history for the text generation.
        temperature (float, optional): The softmax temperature for sampling. Defaults to 0.9.
        max_new_tokens (int, optional): Maximum number of tokens to be generated. Defaults to 256.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
        repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.0.

    Returns:
        Generator[str, None, str]: A generator yielding chunks of generated text.
                                   Returns a final string if an error occurs.
    """

    temperature = max(float(temperature), 1e-2)  # Ensure temperature isn't too low
    top_p = float(top_p)
    
    generate_kwargs = {
        'temperature': temperature,
        'max_tokens': max_new_tokens,
        'top_p': top_p,
        'frequency_penalty': max(-2., min(repetition_penalty, 2.)),
        }

    formatted_prompt = format_prompt(prompt, "openai")

    try:
        stream = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301",
                                                messages=formatted_prompt, 
                                                **generate_kwargs, 
                                                stream=True)
        output = ""
        for chunk in stream:
            output += chunk.choices[0].delta.get("content", "")
            yield output

    except Exception as e:
        if "Too Many Requests" in str(e):
            print("ERROR: Too many requests on OpenAI client")
            gr.Warning("Unfortunately OpenAI is unable to process")
            return "Unfortunately, I am not able to process your request now."
        elif "You didn't provide an API key" in str(e):
            print("Authetification error:", str(e))
            gr.Warning("Authentication error: OpenAI key was either not provided or incorrect")
            return "Authentication error"
        else:
            print("Unhandled Exception:", str(e))
            gr.Warning("Unfortunately OpenAI is unable to process")
            return "I do not know what happened, but I couldn't understand you."
