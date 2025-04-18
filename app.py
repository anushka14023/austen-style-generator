import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Path to your local fine-tuned model directory
model_dir = "fine_tuned_model"

# Check if the model folder exists
if not os.path.exists(model_dir):
    st.error(f"🚫 The model directory '{model_dir}' was not found. Make sure it's uploaded along with app.py.")
    st.stop()

# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

st.set_page_config(page_title="Jane Austen Style Generator", page_icon="✍️")
st.title("✍️ Jane Austen Style Generator")
st.markdown("Generate text in the style of Jane Austen using a fine-tuned GPT-2 model.")

prompt = st.text_area("Enter your prompt:", "It is a truth universally acknowledged,")

if st.button("Generate"):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=250,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    st.markdown("**Generated Text:**")
    st.write(generated_text)
