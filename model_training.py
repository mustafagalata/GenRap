
#-----------------------------------------------------------------
#ATTENTION: I suggest that execute this script on Google Colab !! |
#-----------------------------------------------------------------


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import os

print("Training script started...")
print(f"PyTorch CUDA (GPU) available?: {torch.cuda.is_available()}")

# --- 1. Define Model and Data Paths ---
# This is the base Turkish GPT-2 model we chose
MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2"

# This is the path to our preprocessed "textbook"
DATA_FILE_PATH = "/content/drive/MyDrive/GenRap/train.txt"

# --- 2. Load Training Data ---
if not os.path.exists(DATA_FILE_PATH):
    print(f"ERROR: {DATA_FILE_PATH} not found in Colab runtime!")
    print("Please upload your 'train.txt' file to the left-hand 'Files' panel.")
    raise FileNotFoundError(f"{DATA_FILE_PATH} not found.")

print(f"Loading dataset from '{DATA_FILE_PATH}'...")
# The load_dataset function can read our .txt file directly
raw_dataset = load_dataset("text", data_files={"train": DATA_FILE_PATH})
print(f"Dataset loaded. First 5 lines: {raw_dataset['train'][:5]}")

# --- 3. Load and Configure Tokenizer ---
# The tokenizer translates words/labels (e.g., "[FLEX]") into numbers (e.g., 5012)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 doesn't have a default padding token.
# We'll set it to a new [PAD] token to handle batching.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Tokenizer loaded.")


# --- 4. Preprocess and Tokenize Data ---
# This function applies the tokenizer to a batch of text examples
def tokenize_function(examples):
    # Truncation=True ensures long texts are cut to fit the model's max length
    return tokenizer(examples["text"], truncation=True, padding=False)


print("Tokenizing dataset...")
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # We don't need the raw text column anymore
)

# This is the "page size" or "attention span" of the model
block_size = 512


# This function groups the tokenized texts into blocks of 'block_size'
def group_texts(examples):
    # Concatenate all texts into one giant "scroll"
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the last small chunk to ensure all blocks are the same size
    total_length = (total_length // block_size) * block_size

    # Split the giant scroll into equal-sized "pages" (blocks)
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # For Causal LM (like GPT-2), the model predicts itself.
    # So, the "labels" (what to predict) are the same as the "input_ids" (what it sees).
    result["labels"] = result["input_ids"].copy()
    return result


print("Grouping texts into blocks...")
lm_dataset = tokenized_dataset.map(group_texts, batched=True)
print(f"Processing complete. Total {len(lm_dataset['train'])} blocks created.")

# --- 5. Load the Pre-trained Model ---
print(f"Loading base model ({MODEL_NAME})...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# We must resize the model's token embeddings to include our new [PAD] token
model.resize_token_embeddings(len(tokenizer))
print("Model loaded.")

# --- 6. Define Training Arguments (Hyperparameters) ---
# This is the folder where our fine-tuned model will be saved
OUTPUT_DIR = "genrap-model"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=10,  # Go over the entire dataset (241 songs) x times
    per_device_train_batch_size=4,  # Adjusted batch size for Colab's T4 GPU (16GB VRAM)
    save_steps=500,  # Save a checkpoint every 500 steps
    save_total_limit=2,  # Only keep the last 2 checkpoints (saves disk space)
    logging_steps=100,  # Print the training loss every 100 steps
    fp16=True,  # Use 16-bit precision (mixed precision) for faster training
    report_to="none",
)

# --- 7. Initialize the Trainer ---
# The DataCollator handles creating the batches and applying padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# The Trainer is the main class that manages the entire training loop
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    data_collator=data_collator,
)

# --- 8. Start Training! ---
print("\n" + "=" * 30)
print("STARTING MODEL TRAINING!")
print("This process may take 1-2 hours depending on your dataset size.")
print("Please be patient and monitor the 'Loss' value (it should decrease).")
print("=" * 30 + "\n")

trainer.train()

# --- 9. Save the Final Model ---
print("\n" + "=" * 30)
print("✅ TRAINING COMPLETE! ✅")
print(f"Your fine-tuned model has been saved to the '{OUTPUT_DIR}' directory.")
print("=" * 30)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("All processes finished.")