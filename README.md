# Specialized Finance LLM Chatbot

This repository contains two related implementations:

1. `final_llm_notebook.ipynb` records the Google Colab QLoRA experiment on
   `NousResearch/Llama-2-7b-chat-hf` using two finance instruction datasets.
2. `finetuned_llama2_chatbot.py` is a self-contained CPU-friendly Streamlit
   demonstration. It uses `HuggingFaceTB/SmolLM2-360M-Instruct` with a guarded
   finance system prompt because the development computer cannot host Llama-2 7B.

The local demonstration does not require Flask, ngrok, an API key, or Colab.

## Run locally

```powershell
python -m pip install -r requirements-local.txt
python -m streamlit run finetuned_llama2_chatbot.py
```

On the first run, the model is downloaded from Hugging Face into `E:\hf_cache`.
Later runs use that local cache. Set the `LOCAL_MODEL_CACHE` environment variable
to use another short path. Open `http://localhost:8501` if the browser does not
open automatically.

## Original distributed implementation

The original `flask-ngrok-app.ipynb` loads the large published model and exposes
`POST /generate`; the old Streamlit client called that endpoint. It is retained
as project history, but it is not required by the self-contained local demo.

## Important distinction

Do not claim that the local SmolLM2 deployment is the fine-tuned Llama-2 model.
The notebook is evidence of the Llama-2 QLoRA experiment; the lightweight local
model is a deployment fallback for CPU-only presentation hardware.
