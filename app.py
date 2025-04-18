%%writefile app.py
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("/content/fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("/content/fine_tuned_model")

st.title("✍️ Jane Austen Style Generator")
st.write("Enter a prompt and generate a paragraph in Jane Austen’s style.")

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
            no_repeat_ngram_size=2
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    st.markdown("**Generated Text:**")
    st.write(generated_text)
