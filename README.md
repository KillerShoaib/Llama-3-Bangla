<div align="center"><h1>Llama-3 Bangla</h1></div>
<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65ca6f0098a46a56261ac3ac/O1ATwhQt_9j59CSIylrVS.png" width="300"/>
</div>

---

# Table of Contents
* **Model Weights**
* **Model Details**
* **Pros & Cons**
* **Run the Model**
* **Finetune Script**
* **Why Create Custom Script?**



# Model weights
* [**KillerShoaib/llama-3-8b-bangla-lora**](https://huggingface.co/KillerShoaib/llama-3-8b-bangla-lora)
* [**KillerShoaib/llama-3-8b-bangla-4bit**](https://huggingface.co/KillerShoaib/llama-3-8b-bangla-4bit)
* [**KillerShoaib/llama-3-8b-bangla-GGUF-Q4_K_M**](https://huggingface.co/KillerShoaib/llama-3-8b-bangla-GGUF-Q4_K_M)


# Model Details
* **Base Model:** [unsloth/llama-3-8b-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)
* **Dataset used for finetuning:** [iamshnoo/alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali)
* **Total Epoch:** 2
* **GPU Usage:** Single T4
* **Finetune Package**: [Unsloth](https://github.com/unslothai/unsloth)

# Pros & Cons

## Pros
* Comprehension of the Bangla language including its semantics nuances.
* Model can answer questions correctly if the **context is given**. A perfect use case for **RAG**

## Cons
* Unable to solve **creative** and **logical query**
* Lack of **common knowledge** in Bangla.


# Run the Model

## `FastLanguageModel` from unsloth for 2x faster inference

```python

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "KillerShoaib/llama-3-8b-bangla-lora", ## or KillerShoaib/llama-3-8b-bangla-4bit
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# alpaca_prompt for the model
alpaca_prompt = """Below is an instruction in bangla that describes a task, paired with an input also in bangla that provides further context. Write a response in bangla that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# input with instruction and input
inputs = tokenizer(
[
    alpaca_prompt.format(
        "সুস্থ থাকার তিনটি উপায় বলুন", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

# generating the output and decoding it
outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
tokenizer.batch_decode(outputs)

```

**Output:**
```
>>> ["<|begin_of_text|>Below is an instruction in bangla that describes a task, paired with an input also in bangla that provides further context. Write a response in bangla that appropriately completes the request.\n\n### Instruction:\nসুস্থ থাকার তিনটি উপায় বলুন\n\n### Input:\n\n\n### Response:\n১. নিয়মিত ব্যায়াম করুনঃ নিয়মিত শারীরিক ক্রিয়াকলাপ স্বাস্থ্যকর হওয়ার একটি গুরুত্বপূর্ণ অংশ। এটি শরীরের ওজন হ্রাস করতে, হৃদরোগের স্বাস্থ্য উন্নত করতে এবং শক্তির মাত্রা বাড়াতে সহায়তা করতে পারে। প্রতিদিন কমপক্ষে ৩০ মিনিট মাঝারি তীব্রতার ব্যায়াম করা একটি সুস্থ জীবনযাত্রার জন্য অপরিহার্য। ২. ভারসাম্যপূর্ণ খাদ্য খানঃ একটি সুস্থ জীবনযাত্রার জন্য ভারসাম্যপূর্ণ খাদ্য খাওয়া অত্যন্ত গুরুত্বপূর্ণ। এর অর্থ হ'ল ফল, সবজি, পূর্ণ শস্য, চর্বিযুক্ত প্রোটিন এবং স্বাস্থ্যকর ফ্যাট সহ বিভিন্ন খাদ্য গোষ্ঠী খাওয়া। এটি শরীরকে সঠিকভাবে কাজ করার জন্য প্রয়োজনীয় পুষ্টি সরবরাহ করে এবং স্থূলতা, ডায়াবেটিস এবং হৃদরোগের মতো দীর্ঘস্থায়ী রোগের ঝুঁকি হ্রাস করে। ৩. পর্যাপ্ত ঘুম পানঃ ঘুম শারীরিক ও মানসিক স্বাস্থ্যের জন্য অপরিহার্য। প্রতি রাতে পর্যাপ্ত ঘুম পাওয়া একটি সুস্থ জীবনযাত্রার জন্য অপরিহার্য। এটি শরীরকে পুনরুদ্ধার করতে, প্রতিরোধ ক্ষমতা বাড়াতে এবং মানসিক স্বাস্থ্যের উন্নতি করতে সহায়তা করে। প্রতিদিন পর্যাপ্ত ঘুম পাওয়া এবং একটি স্বাস্থ্যকর ঘুমের প্রোগ্রাম বজায় রাখা গুরুত্বপূর্ণ।<|end_of_text|>"]
```

## `AutoModelForPeftCausalLM` from Hugginface (only for [**LoRA**](https://huggingface.co/KillerShoaib/llama-3-8b-bangla-4bit) adapter model)

```python

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    "KillerShoaib/llama-3-8b-bangla-lora",
    load_in_4bit = True,
)
tokenizer = AutoTokenizer.from_pretrained("KillerShoaib/llama3-8b-4bit-bangla")

alpaca_prompt = """Below is an instruction in bangla that describes a task, paired with an input also in bangla that provides further context. Write a response in bangla that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
[
    alpaca_prompt.format(
        "সুস্থ থাকার তিনটি উপায় বলুন", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)
tokenizer.batch_decode(outputs)
```

## `AutoModelForCausalLM` from Hugginface (for [**4 bit** ](https://huggingface.co/KillerShoaib/llama-3-8b-bangla-4bit) model)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "KillerShoaib/llama-3-8b-bangla-4bit"
tokenizer_name = model_name

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)


# Text prompt to start generation
alpaca_prompt = """Below is an instruction in bangla that describes a task, paired with an input also in bangla that provides further context. Write a response in bangla that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Encode the prompt text
inputs = tokenizer(
[
    alpaca_prompt.format(
        "x পরিবর্তনশীল 4x + 2y = 10 হিসাবে সংজ্ঞায়িত করা হয়। x এর মান খুঁজুন।", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

# output
outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)

tokenizer.batch_decode(outputs)
```


# Finetuning Script
* **Original Unsloth Script: [Alpaca + Llama-3 8b full example](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)**
* **My Own Custom Script: [Llama-3 8b Finetuing Guide on Bangla Alpaca Dataset](https://github.com/KillerShoaib/Llama-3-Bangla/blob/main/Llama_3_8b_Finetuing_Guide_on_Bangla_Alpaca_Dataset.ipynb)**



# Why Create Custom Script?
- Original script doesn't show how to handle **incremental training**. (start training from a checkpoint)
- I was using Colab and Kaggle back & forth. I trained the model for **2 epochs (~80hrs)**. I've to finetuned the model in such a way that my script always ran **less than 12hrs**.
- I've utilized [**Weights & Biases**](https://wandb.ai/) for tracking and saving the model checkpoint.
- In that script, I've shown how one can finetune the llama-3 8b model in **Kaggle** or **Colab** with incremental training using `max_steps` instead of `num_train_epochs` and continue the training from previous checkpoints, instead of training all at once.

