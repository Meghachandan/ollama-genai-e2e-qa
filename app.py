import os
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# LangSmith Tracking (Optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer the user's questions accurately and concisely.",
        ),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question, model_name, temperature):
    llm = OllamaLLM(
        model=model_name,
        temperature=temperature,
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain.invoke({"question": question})


# Streamlit UI
st.set_page_config(page_title="Ollama Q&A Chatbot", page_icon="🤖")

st.title("🤖 Ollama End-to-End GenAI Q&A Chatbot")

st.write("Ask any question using your locally running Ollama model.")

# Sidebar
st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Choose an Ollama Model",
    [
        "mistral",
        "llama3",
        "gemma",
        "phi3",
    ],
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
)

# User Input
user_input = st.text_input("Enter your question")

if st.button("Generate Response"):
    if user_input.strip():

        with st.spinner("Generating response..."):
            response = generate_response(
                user_input,
                model_name,
                temperature,
            )

        st.success(response)

    else:
        st.warning("Please enter a question.")

