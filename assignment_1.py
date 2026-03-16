""" Natural Language Processing : Assignment 1 """
# Studant Name: (John) Paul Nagle
# Student ID:   R00065426

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {'GPU ✓' if torch.cuda.is_available() else 'CPU'}")


# ── Exercise 1 ────────────────────────────────────────────────────────────────
# TODO: Load the 'distilgpt2' tokenizer and causal LM model.
#       Set the pad_token to eos_token.
#       Print the model name, number of parameters, and vocab size.
# Hint: use AutoTokenizer.from_pretrained() and AutoModelForCausalLM.from_pretrained()
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "distilgpt2"

# YOUR CODE HERE
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print(f"Model    : {MODEL_NAME}")
print(f"Params   : {model.num_parameters():,}")
print(f"Vocab sz : {tokenizer.vocab_size:,}")

# ── Exercise 1b ───────────────────────────────────────────────────────────────
# TODO: Tokenise the sentence below.
#       1. Print the list of token IDs
#       2. Decode the token IDs back to a string
#       3. Print the individual token strings (subword pieces)
# Hint: tokenizer.encode(), tokenizer.decode(), tokenizer.convert_ids_to_tokens()
# ─────────────────────────────────────────────────────────────────────────────

text = "Transformers are amazing for NLP tasks!"

# YOUR CODE HERE
# 1. Print the list of token IDs
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# 2. Decode the token IDs back to a string
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded text: {decoded_text}")

# 3. Print the individual token strings (subword pieces)
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(f"Individual tokens: {tokens}")

# ── Exercise 2a ───────────────────────────────────────────────────────────────
# TODO: Complete the generate_text() function below.
#       It should support four strategies: 'greedy', 'beam', 'temperature', 'top_p'
#       Return ONLY the newly generated text (not the prompt).
# Hint: use model.generate(); decode only tokens after the prompt length
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(prompt: str, max_new_tokens: int = 80, strategy: str = "greedy") -> str:
    inputs = tokenizer(prompt, return_tensors="pt")

    # Build gen_kwargs based on strategy
    if strategy == "greedy":
        gen_kwargs = {"do_sample": False}
    elif strategy == "beam":
        gen_kwargs = {"do_sample": False, "num_beams": 5}
    elif strategy == "temperature":
        gen_kwargs = {"do_sample": True, "temperature": 0.8}
    elif strategy == "top_p":
        gen_kwargs = {"do_sample": True, "top_p": 0.9, "temperature": 0.7}

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, **gen_kwargs)

    # Extract newly generated tokens (after prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]

    response = tokenizer.decode(new_tokens)
    return response.strip()


# Quick test
print(generate_text("Deep learning is", strategy="greedy"))

# ── Exercise 2b ───────────────────────────────────────────────────────────────
# TODO: Using the generate_text() function you wrote above,
#       run all four strategies on PROMPT and print the results.
#       Observe: which strategy gives the most creative output?
#                which gives the most repetitive?
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = "Artificial intelligence is transforming the world by"

# YOUR CODE HERE
strategies = ["greedy", "beam", "temperature", "top_p"]

print(f"\"{PROMPT}...\" \n")
for strategy in strategies:
    print(f"Strategy: {strategy.upper()}")
    print(f"Output: {generate_text(PROMPT, strategy=strategy)}")
    print("--------------------------------------------------------------------------------")
