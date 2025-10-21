import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import os

# NOTE: For this to work in deployment, you MUST copy the 'retnet.py' 
# file from the cloned repository's 'src' directory into your project root.
try:
    from retnet import RetNet
except ImportError:
    st.error("FATAL: The custom 'retnet.py' module is missing. Please copy it into the app's root directory.")
    st.stop()


# --- 1. Hugging Face Setup and Constants ---
HF_MODEL_REPO_ID = "sebastiancgeorge/Retnet_coder" 
MODEL_FILE = "model.pt" 
MAX_LEN = 128
HIDDEN_DIM = 512
HEADS = 8
FFN_SIZE = HIDDEN_DIM * 4
LAYERS = 6


# --- 2. Model Loading and Caching ---

# @st.cache_resource is essential to prevent reloading the model on every interaction
@st.cache_resource
def load_model_and_tokenizer(model_repo_id):
    """Loads all model components from the Hugging Face Hub."""
    
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = len(tokenizer)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None, None, None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Model Components
    embedding = nn.Embedding(vocab_size, HIDDEN_DIM)
    output_proj = nn.Linear(HIDDEN_DIM, vocab_size, bias=False)
    output_proj.weight = embedding.weight # Tie weights

    model = RetNet(
        layers=LAYERS,
        hidden_dim=HIDDEN_DIM,
        ffn_size=FFN_SIZE,
        heads=HEADS,
        double_v_dim=False
    )
    
    # Download and Load Checkpoint
    try:
        st.info(f"Downloading model weights from {model_repo_id}...")
        model_path = hf_hub_download(repo_id=model_repo_id, filename=MODEL_FILE)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state'])
        embedding.load_state_dict(checkpoint['embedding_state'])
        output_proj.load_state_dict(checkpoint['output_proj_state'])
        
        model.to(device).eval()
        embedding.to(device).eval()
        output_proj.to(device).eval()
        st.success("Model loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading checkpoint from Hugging Face Hub. Error: {e}")
        return None, None, None, None, None

    return model, embedding, output_proj, tokenizer, device


# --- 3. Inference Function (Code Generation) ---

def generate_code(prompt, model, embedding, output_proj, tokenizer, device, max_length=50, temperature=0.8, top_k=50):
    """Generates code completion using the RetNet model."""
    
    # Set to eval mode again for safety
    model.eval()
    
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN//2)
    input_ids = tokens["input_ids"].to(device)
    
    generated = input_ids.clone()
    prompt_length = generated.shape[1]
    
    with torch.no_grad():
        for _ in range(max_length):
            current_input = generated if generated.shape[1] <= MAX_LEN else generated[:, -MAX_LEN:]
            
            embeds = embedding(current_input)
            outputs = model(embeds) # Parallel forward pass for generation
            
            logits = output_proj(outputs[:, -1, :])
            
            # Sampling
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[..., -1, None]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() == 0:
                break
                
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
            
            generated = torch.cat([generated, next_token], dim=1)
    
    generated_tokens = generated[0, prompt_length:]
    if len(generated_tokens) == 0:
        return "[Model generated nothing new]"
    
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text.strip() if text.strip() else "[Empty generation]"


# --- 4. Streamlit UI Layout ---
st.set_page_config(page_title="RetNet Code Generation Demo", layout="wide")

st.title("🐍 RetNet Code Generation Demo")
st.markdown(f"""
Code generator powered by the Retentive Network model: **`{HF_MODEL_REPO_ID}`**.
""")

# Load model components
model, embedding, output_proj, tokenizer, device = load_model_and_tokenizer(HF_MODEL_REPO_ID)

if model is None:
    st.stop()

# Sidebar for Parameters
st.sidebar.header("Generation Parameters")
max_len = st.sidebar.slider("Max Length", min_value=10, max_value=256, value=60, step=10)
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=0.75, step=0.05)
top_k = st.sidebar.slider("Top K Sampling", min_value=0, max_value=100, value=50, step=5)
st.sidebar.caption(f"Running on device: **{device}**")

# Main Input Area
st.subheader("Enter your Code Prompt:")
prompt = st.text_area(
    "Code Prompt",
    value="def compute_mean_average(data_list):\n    # Calculate the sum of all elements in the list\n",
    height=150,
    key="prompt_input"
)

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a code prompt to begin generation.")
    else:
        st.divider()
        st.markdown("### 🤖 Generated Code:")
        with st.spinner("Generating..."):
            try:
                result = generate_code(
                    prompt, 
                    model, 
                    embedding, 
                    output_proj, 
                    tokenizer, 
                    device, 
                    max_length=max_len, 
                    temperature=temperature, 
                    top_k=top_k
                )
                
                # Display the prompt and the generated completion
                full_output = prompt + result
                st.code(full_output, language='python')
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

st.divider()
st.caption(f"Model: {HF_MODEL_REPO_ID}")
