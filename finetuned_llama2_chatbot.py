import os

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")
MODEL_CACHE = os.getenv("LOCAL_MODEL_CACHE", "E:/hf_cache")
SYSTEM_PROMPT = """You are a careful finance education assistant.
Explain financial terms in simple language and use short examples when useful.
Do not invent current prices, laws, or market facts. If current data is required,
say that you do not have live market data. Do not give personalized investment,
tax, or legal advice. State uncertainty clearly."""

st.set_page_config(page_title="Finance LLM Chatbot", page_icon="💬")
st.title("💬 Finance LLM Chatbot")
st.caption("Self-contained local demonstration • educational answers only")


@st.cache_resource(show_spinner="Loading the local language model for the first time...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=MODEL_CACHE,
    )
    model.eval()
    return model, tokenizer


def generate_response(question, history):
    model, tokenizer = load_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in history[-4:]:
        messages.append({"role": message["role"], "content": message["content"]})
    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


with st.sidebar:
    st.subheader("Demo information")
    st.write(f"Local deployment model: `{MODEL_NAME}`")
    st.info(
        "The Colab notebook records the separate Llama-2 QLoRA experiment. "
        "This smaller model is used for a reliable CPU-only live demonstration."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.chat_message("assistant").write(
        "Hello! Ask me to explain a finance concept, ratio, or general scenario."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask a finance question"):
    previous_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating locally on CPU..."):
                answer = generate_response(question, previous_messages)
        except Exception as exc:
            st.error(
                "The local model could not start. Check the internet connection on "
                "the first run and install requirements-local.txt."
            )
            st.exception(exc)
        else:
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("This prototype provides educational information, not financial advice.")
