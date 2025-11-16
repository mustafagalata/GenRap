import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Load the Fine-Tuned Model and Tokenizer ---

MODEL_DIR = "genrap-model"

print(f"Loading trained model from '{MODEL_DIR}'...")
print("This may take 1-2 minutes depending on the model size...")

try:

    # Load our fine-tuned model and its specific tokenizer from disk
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

except EnvironmentError:

    print(f"ERROR: Directory '{MODEL_DIR}' not found.")
    exit()

print("✅ Model successfully loaded!")

# --- 2. Set the Device (CPU or GPU) ---

# Check if a CUDA-enabled GPU (NVIDIA) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model will run on: '{device}'")

# ==========================================================
# 🎵 CREATIVE PLAYGROUND (YOUR SETTINGS) 🎵
# ==========================================================

# 1. Prompt: Tell the model which labels to start with

# CLASS = [["DUYGUSAL"], ["AGRESİF"], ["EĞLENCELİ"], ["SOKAK"], ["FELSEFİK"], ["FLEX"], ["BATTLE"]]

prompt_text = "[FELSEFİK] [DUYGUSAL]"

# 2. Length: The maximum total length (in tokens) of the generated text
max_length = 250  # Approx 150-200 words (a verse + chorus)

# ==========================================================

print("\n" + "=" * 30)
print(f"Using Prompt: '{prompt_text}'")
print(f"Generating lyrics up to {max_length} tokens long...")
print("=" * 30 + "\n")

# --- 3. Tokenize the Prompt ---

# Convert our text prompt into numbers (token IDs) that the model understands
# return_tensors="pt" -> returns PyTorch tensors
inputs = tokenizer(prompt_text, return_tensors="pt")

# Send the input tokens to the correct device (CPU or GPU)
inputs = inputs.to(device)

# --- 4. Generate Lyrics ---

with torch.no_grad():

    # model.generate() creates new tokens based on our prompt
    output_sequences = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,  # 1. Max length setting
        num_return_sequences=1,  # Only generate one version

        # --- CREATIVITY SETTINGS ---
        do_sample=True,  # MUST be True for creative text. (False=boring, repetitive text)
        temperature=0.95,  # Creativity/Chaos (Lower=safer, Higher=more random/nonsensical)
        top_k=50,  # Narrows the random choice pool to the top 50 most likely words
        top_p=0.95,  # Nucleus sampling (further refines the choice pool)
        pad_token_id=tokenizer.eos_token_id,  # Set the padding token to the End-Of-Sentence token
        repetition_penalty=1.3
    )


# --- 5. Decode the Output ---

# Get the raw list of token IDs (numbers) from the output
output_ids = output_sequences[0]

generated_text = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

# We must decode the original prompt with the EXACT same settings
prompt_text = tokenizer.decode(
    inputs.input_ids[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

# And then slice it off the beginning of the generated text
generated_text = generated_text[len(prompt_text):]

if tokenizer.pad_token:
    generated_text = generated_text.replace(tokenizer.pad_token, "\n")

if tokenizer.eos_token:
    generated_text = generated_text.replace(tokenizer.eos_token, "\n")


print("--- GENERATED LYRICS ---")
print(generated_text.strip())
print("------------------------")