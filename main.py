import streamlit as st
from llama_cpp import Llama

# --------------------------
# Load GGUF Model
# --------------------------
@st.cache_resource
def load_model():
    llm = Llama(
        model_path="unsloth.Q4_K_M.gguf",  # Change to your GGUF model path
        n_ctx=4096,
        n_threads=8,       # Adjust based on CPU cores
        n_gpu_layers=-1    # Use GPU acceleration if available
    )
    return llm

llm = load_model()

# --------------------------
# Streamlit UI
# --------------------------
st.title("GGUF Model Summarizer")
st.write("Enter a conversation or text, and get a summary.")

prompt = st.text_area("Enter your text:", height=150)

if st.button("Generate Summary") and prompt.strip():
    with st.spinner("Generating summary..."):
        # Minimal instruction: just ask to summarize
        instruction = f"Summarize the following conversation in a clear, concise way:\n\n{prompt}\n\nSummary:"
        
        output = llm(
            instruction,
            max_tokens=512,      # Allow multiple sentences in the summary
            temperature=0.7,
            stop=["###", "\n\n\n"]
        )
        
        response_text = output["choices"][0]["text"].strip()

    st.subheader("Summary:")
    st.write(response_text)
