# %% [markdown]
# # NLP  Lab Week 7-8 
# # Open Source LLMs
# ## Fine-Tuning · Text Generation · Question Answering
# 
# **Prerequisites:** Python 3.8+, ~4 GB free disk space  
# **Estimated time:** 90–120 minutes  
# 
# ---
# ### Learning Objectives
# By the end of this lab you will be able to:
# 1. Load and run inference with small open-source LLMs
# 2. Compare text-generation decoding strategies
# 3. Build extractive and generative QA pipelines
# 4. Fine-tune a causal LM on custom data using the HuggingFace `Trainer`
# 5. Apply LoRA for parameter-efficient fine-tuning
# 
# ---
# ### Models used (all < 500 MB, CPU-friendly)
# | Model | Size | Task |
# |---|---|---|
# | `distilgpt2` | ~330 MB | Causal LM / text generation |
# | `google/flan-t5-small` | ~300 MB | Seq2Seq / instruction QA |
# | `deepset/roberta-base-squad2` | ~500 MB | Extractive QA |
# 
# %% [markdown]
# ##  Setup — Install Dependencies
# %%
# Run this cell once to install all required packages
# !pip install transformers datasets peft accelerate torch sentencepiece -q
# %% [markdown]
# ##  Imports
# %%
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

# %% [markdown]
# ---
# ## Section 1 — Load a Model and Tokenizer
# 
# The HuggingFace `transformers` library provides a unified API for hundreds of models.
# 
# **Key classes:**
# - `AutoTokenizer.from_pretrained(name)` — loads the correct tokenizer for a model
# - `AutoModelForCausalLM.from_pretrained(name)` — loads a causal language model (GPT family)
# 
# > 💡 **Causal LM** = predicts the *next* token given all previous tokens. Used for text generation.
# 
# %%
# ── Exercise 1 ────────────────────────────────────────────────────────────────
# TODO: Load the 'distilgpt2' tokenizer and causal LM model.
#       Set the pad_token to eos_token.
#       Print the model name, number of parameters, and vocab size.
# Hint: use AutoTokenizer.from_pretrained() and AutoModelForCausalLM.from_pretrained()
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "distilgpt2"8

# YOUR CODE HERE
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print(f"Model    : {MODEL_NAME}")
print(f"Params   : {model.num_parameters():,}")
print(f"Vocab sz : {tokenizer.vocab_size:,}")

# %% [markdown]
# ### 1b — Tokenisation
# 
# Tokenisation converts raw text into integer IDs that the model understands.
# 
# %%
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
# %% [markdown]
# ---
# ## Section 2 — Text Generation & Decoding Strategies
# 
# `model.generate()` supports multiple decoding strategies:
# 
# | Strategy | `do_sample` | Extra args | Character |
# |---|---|---|---|
# | **Greedy** | `False` | — | Deterministic, repetitive |
# | **Beam search** | `False` | `num_beams=5` | Better quality, slower |
# | **Temperature** | `True` | `temperature=0.8` | Creative / risky |
# | **Top-p (nucleus)** | `True` | `top_p=0.9, temperature=0.7` | Best diversity/quality balance |
# 
# %% [markdown]
# ### 2a — Helper function
# %%
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

# %% [markdown]
# ### 2b — Compare all strategies on the same prompt
# %%
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
# %% [markdown]
# ---
# ## Section 3 — Question Answering with FLAN-T5
# 
# `google/flan-t5-small` is a Seq2Seq model fine-tuned to follow natural language instructions.
# It can answer open-domain questions, translate, summarise, and classify — all in one model.
# 
# > 💡 **Seq2Seq** = Encoder processes input, Decoder generates output token by token.
# 
# %%
# ── Exercise 3a ───────────────────────────────────────────────────────────────
# TODO: Load the tokenizer and Seq2Seq model for 'google/flan-t5-small'.
#       Use AutoTokenizer and AutoModelForSeq2SeqLM.
#       Print the number of parameters.
# ─────────────────────────────────────────────────────────────────────────────

T5_NAME = "google/flan-t5-small"

# YOUR CODE HERE

tok_t5 = AutoTokenizer.from_pretrained(T5_NAME)
mdl_t5 = AutoModelForSeq2SeqLM.from_pretrained(T5_NAME)

print(f"Model    : {T5_NAME}")
print(f"Params   : {mdl_t5.num_parameters():,}")
print(f"Vocab sz : {tok_t5.vocab_size:,}")

# %% [markdown]
# ### 3b — Inference function
# %%
# ── Exercise 3b ───────────────────────────────────────────────────────────────
# TODO: Write ask_flan_t5(instruction, max_new_tokens=100).
#       Tokenise the instruction, run mdl_t5.generate(), decode and return the answer.
#
# Then run it on the four questions below and observe the outputs.
# ─────────────────────────────────────────────────────────────────────────────

def ask_flan_t5(instruction: str, max_new_tokens: int = 100) -> str:
    inputs = tok_t5(instruction, return_tensors="pt")

    output_ids = mdl_t5.generate(**inputs, max_new_tokens=max_new_tokens)

    response = tok_t5.decode(output_ids[0], skip_special_tokens=True)
    return response.strip()


questions = [
    "What is the capital of Japan?",
    "Translate to French: The weather is beautiful today.",
    "Classify the sentiment of: I absolutely loved the movie!",
    "What are two advantages of renewable energy?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask_flan_t5(q)}\n")

# %% [markdown]
# ---
# ## Section 4 — Extractive Question Answering
# 
# Extractive QA finds the answer *span* inside a provided context.  
# Model: `deepset/roberta-base-squad2` — fine-tuned on SQuAD2 dataset.
# 
# ```
# Context : "The Transformer was introduced in 2017 by Vaswani et al."
# Question: "Who introduced the Transformer?"
# Answer  : "Vaswani et al."  ← a span extracted from the context
# ```
# 
# %%
# ── Exercise 4 ────────────────────────────────────────────────────────────────
# TODO:
#   1. Create a QA pipeline using pipeline("question-answering",
#      model="deepset/roberta-base-squad2")
#   2. For each question in qa_pairs, run the pipeline on the provided context
#   3. Print the answer and the confidence score
# Hint: qa_pipe(question=..., context=...)
# ─────────────────────────────────────────────────────────────────────────────

context = """
The Transformer architecture was introduced in the landmark 2017 paper
'Attention Is All You Need' by Vaswani et al. at Google Brain. Unlike
recurrent neural networks, Transformers process all tokens in parallel
using self-attention, which allows each token to attend to every other
token in the sequence. BERT, GPT, and T5 are all based on Transformers.
"""

qa_pairs = [
    "When was the Transformer architecture introduced?",
    "What mechanism do Transformers use to process tokens?",
    "Which models are based on the Transformer architecture?",
]

# YOUR CODE HERE — create pipeline and run inference

# %% [markdown]
# ---
# ## Section 5 — Fine-Tuning a Causal LM on Custom Data
# 
# Fine-tuning adapts a pre-trained model to your domain by continuing training on new data.
# 
# **Steps:**
# 1. Prepare a dataset of domain texts
# 2. Tokenise (labels = input_ids for causal LM)
# 3. Configure `TrainingArguments`
# 4. Train with `Trainer`
# 5. Save and reload
# 
# %% [markdown]
# ### 5a — Prepare the dataset
# %%
# ── Exercise 5a ───────────────────────────────────────────────────────────────
# TODO:
#   1. Wrap CUSTOM_TEXTS in a HuggingFace Dataset using Dataset.from_dict()
#   2. Write a tokenize_fn that:
#        - tokenises the 'text' column with max_length=64 and padding
#        - sets labels = input_ids (self-supervised causal LM objective)
#   3. Apply the function with dataset.map(tokenize_fn, batched=True)
#   4. Set format to "torch"
#   5. Print the number of samples and column names
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_TEXTS = [
    "Machine learning models learn patterns from data to make predictions.",
    "Neural networks consist of layers of interconnected nodes called neurons.",
    "Gradient descent minimises the loss function by updating model weights iteratively.",
    "Overfitting occurs when a model performs well on training data but poorly on new data.",
    "Regularisation techniques like dropout help prevent overfitting in deep networks.",
    "Transfer learning allows models to apply knowledge from one task to another.",
    "Attention mechanisms allow models to focus on the most relevant parts of the input.",
    "BERT uses bidirectional context to build rich language representations.",
    "The learning rate controls the size of update steps during gradient descent.",
    "Batch normalisation stabilises training by normalising layer activations.",
    "Convolutional neural networks are highly effective for image recognition tasks.",
    "Large language models are pre-trained on massive text corpora before fine-tuning.",
    "Tokenisation converts raw text into numerical tokens the model can process.",
    "An embedding layer maps discrete tokens to continuous high-dimensional vectors.",
    "The softmax function converts raw logits into a probability distribution over tokens.",
]

FT_NAME = "distilgpt2"
ft_tok  = AutoTokenizer.from_pretrained(FT_NAME)
ft_tok.pad_token = ft_tok.eos_token

# YOUR CODE HERE
def tokenize_fn(examples):
    pass

raw_ds = ...
tok_ds = ...

# %% [markdown]
# ### 5b — Configure and run training
# %%
# ── Exercise 5b ───────────────────────────────────────────────────────────────
# TODO:
#   1. Load a fresh distilgpt2 model with AutoModelForCausalLM
#   2. Create TrainingArguments:
#        - output_dir="./finetuned_llm", num_train_epochs=3
#        - per_device_train_batch_size=4, learning_rate=5e-5
#        - report_to="none"
#   3. Create a DataCollatorForLanguageModeling with mlm=False
#   4. Create a Trainer and call trainer.train()
# ─────────────────────────────────────────────────────────────────────────────

# YOUR CODE HERE
ft_model = ...

training_args = TrainingArguments(
    # fill in arguments
)

data_collator = ...
trainer = ...
trainer.train()

# %% [markdown]
# ### 5c — Save and reload
# %%
# ── Exercise 5c ───────────────────────────────────────────────────────────────
# TODO:
#   1. Save the fine-tuned model and tokenizer to "./finetuned_llm/final"
#      using trainer.save_model() and ft_tok.save_pretrained()
#   2. Reload the model with AutoModelForCausalLM.from_pretrained()
#   3. Write generate_ft(prompt) that generates text with top-p sampling
#   4. Test on the two prompts below
# ─────────────────────────────────────────────────────────────────────────────

SAVE_PATH = "./finetuned_llm/final"

# YOUR CODE HERE

def generate_ft(prompt, max_new_tokens=60):
    # YOUR CODE HERE
    pass

for prompt in ["Neural networks consist of", "The learning rate controls"]:
    print(f"PROMPT : {prompt}")
    print(f"OUTPUT : {generate_ft(prompt)}\n")

# %% [markdown]
# ---
# ## Section 6 — LoRA: Parameter-Efficient Fine-Tuning
# 
# **Problem:** Fine-tuning large models requires updating hundreds of millions of weights → slow and memory-hungry.  
# **Solution:** LoRA (Low-Rank Adaptation) inserts small trainable matrices **A** and **B** into selected layers:
# 
# ```
#   h = W₀x  +  ΔWx         (standard fine-tuning)
#   h = W₀x  +  BAx          (LoRA: B and A are small, W₀ is frozen)
# ```
# 
# With `r=8`, LoRA trains **< 0.5%** of parameters while achieving comparable performance.
# 
# %%
# ── Exercise 6 ────────────────────────────────────────────────────────────────
# TODO:
#   1. Import LoraConfig, get_peft_model, TaskType from peft
#   2. Load a fresh distilgpt2 base model
#   3. Create a LoraConfig with:
#        task_type=CAUSAL_LM, r=8, lora_alpha=32,
#        target_modules=["c_attn"], lora_dropout=0.1
#   4. Wrap the base model with get_peft_model(base, lora_config)
#   5. Print total params, trainable params, and trainable %
#
# Question to think about: why is the trainable % so small?
# ─────────────────────────────────────────────────────────────────────────────

# YOUR CODE HERE

# %% [markdown]
# ---
# ## Section 7 — Pipelines: One-Liner Inference
# 
# HuggingFace `pipeline()` wraps tokenisation + inference + decoding into a single object.
# 
# %%
# ── Exercise 7 ────────────────────────────────────────────────────────────────
# TODO:
#   1. Create a text-generation pipeline with model="distilgpt2"
#      and generate a completion for "Once upon a time"
#   2. Create a sentiment-analysis pipeline (default model is fine)
#      and classify: "This lab is incredibly well-structured!"
#   3. (Bonus) Try a translation pipeline for EN→FR
# ─────────────────────────────────────────────────────────────────────────────

# YOUR CODE HERE

# %% [markdown]
# ---
# ## Section 8 — Reflection Questions
# 
# Answer these in the markdown cells below (double-click to edit).
# 
# %% [markdown]
# **Q1.** What is the difference between a Causal LM and a Seq2Seq model? Give one use-case for each.
# %% [markdown]
# *Your answer here...*
# %% [markdown]
# **Q2.** Explain in your own words why Top-p sampling often produces better text than greedy decoding.
# %% [markdown]
# *Your answer here...*
# %% [markdown]
# **Q3.** When would you choose extractive QA over generative QA?
# %% [markdown]
# *Your answer here...*
# %% [markdown]
# **Q4.** What is LoRA and why is it useful for fine-tuning large models on limited hardware?
# %% [markdown]
# *Your answer here...*
# %% [markdown]
# **Q5.** You have 500 customer support tickets. Describe the steps to fine-tune a model to auto-reply to them.
# %% [markdown]
# *Your answer here...*
# %% [markdown]
# 
# 
# ### Next Steps
# - Explore more models at [huggingface.co/models](https://huggingface.co/models)
# - Try fine-tuning on a real dataset: `load_dataset("imdb")` or `load_dataset("squad")`
# - Scale up with **Mistral-7B** or **LLaMA-3** using LoRA on Google Colab (free GPU)
# - Add evaluation: `pip install evaluate` → BLEU, ROUGE, F1
# - Deploy: wrap your model in FastAPI for a live QA endpoint
# 