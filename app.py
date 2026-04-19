import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==============================
# Load environment variables
# ==============================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="AI Document Classifier", layout="wide")

st.title("🤖 AI Office Document Classification System")
st.write("Upload documents and let AI classify and route them.")

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt"],   # ✅ added txt
    accept_multiple_files=True
)

# ==============================
# Initialize LLM once
# ==============================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an office document classification assistant.

Classify the following document into ONE of:
Invoice, Purchase Order, Contract, HR Document, Internal Memo, Financial Report.

Return STRICT JSON with:
- document_type
- confidence (0 to 1)
- recommended_department
- reasoning (1 short sentence)

Document content:
----------------
{document_text}
""")

chain = prompt | llm

# ==============================
# Process uploaded files
# ==============================
if uploaded_files:

    for i, uploaded_file in enumerate(uploaded_files):

        st.divider()
        st.subheader(f"📄 Document {i+1}: {uploaded_file.name}")

        # Detect extension
        file_extension = uploaded_file.name.split(".")[-1].lower()
        temp_file = f"temp_{i}.{file_extension}"

        # Save file
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())

        # ==============================
        # Select Loader
        # ==============================
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file)

        elif file_extension == "docx":
            loader = Docx2txtLoader(temp_file)

        elif file_extension == "txt":
            loader = TextLoader(temp_file, encoding="utf-8")

        else:
            st.error("Unsupported file type")
            continue

        documents = loader.load()

        # ==============================
        # Split text
        # ==============================
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        split_document = text_splitter.split_documents(documents)

        document_text = "\n\n".join([doc.page_content for doc in split_document])

        # Limit size for LLM
        document_text = document_text[:6000]

        # ==============================
        # LLM Call
        # ==============================
        with st.spinner("Analyzing document..."):
            response = chain.invoke({
                "document_text": document_text
            })

        try:
            result = json.loads(response.content)

            st.markdown(f"**Document Type:** {result['document_type']}")
            st.markdown(f"**Recommended Department:** {result['recommended_department']}")
            st.markdown(f"**Reasoning:** {result['reasoning']}")

            confidence = float(result["confidence"])

            st.progress(confidence)
            st.write(f"Confidence Score: {confidence:.2f}")

            if confidence >= 0.85:
                st.success("High confidence")
            elif confidence >= 0.60:
                st.warning("Medium confidence")
            else:
                st.error("Low confidence")

        except:
            st.error("LLM output was not valid JSON.")
            st.write(response.content)
