from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./austen_model")
tokenizer = GPT2Tokenizer.from_pretrained("./austen_model")

# Set the prompt you want to start with
input_prompt = "It is a truth universally acknowledged"

# Encode the prompt
inputs = tokenizer.encode(input_prompt, return_tensors='pt')

# Generate the output
outputs = model.generate(
    inputs,
    max_length=150,  # Adjust the length of the generated text
    num_return_sequences=1,  # Number of generated outputs
    temperature=0.9,  # Controls randomness (lower = more deterministic)
    top_k=50,  # Limit sampling to the top-k words
    top_p=0.95,  # Nucleus sampling
    do_sample=True,  # Ensure sampling
    pad_token_id=tokenizer.eos_token_id  # Prevent issues with padding tokens
)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
