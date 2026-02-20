import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
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

st.title("📄 AI Office Document Classification System")
st.write("Upload a document and let AI classify and route it.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # ==============================
    # Load PDF
    # ==============================
    loader = PyPDFLoader("temp.pdf")
    pdf_documents = loader.load()

    # ==============================
    # Split Text
    # ==============================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_document = text_splitter.split_documents(pdf_documents)

    document_text = "\n\n".join([doc.page_content for doc in split_document])

    # ==============================
    # Initialize Groq LLM
    # ==============================
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    # ==============================
    # Prompt
    # ==============================
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

    with st.spinner("Analyzing document..."):
        response = chain.invoke({
            "document_text": document_text
        })

    try:
        result = json.loads(response.content)

        st.subheader("📊 Classification Result")

        st.markdown(f"**Document Type:** {result['document_type']}")
        st.markdown(f"**Recommended Department:** {result['recommended_department']}")
        st.markdown(f"**Reasoning:** {result['reasoning']}")

        confidence = float(result["confidence"])

        st.progress(confidence)
        st.write(f"Confidence Score: {confidence:.2f}")

        # Routing Logic
        if confidence >= 0.85:
            st.success("✅ High confidence — Ready for auto-routing.")
        elif confidence >= 0.60:
            st.warning("⚠️ Medium confidence — Please review before routing.")
        else:
            st.error("❌ Low confidence — Manual review recommended.")

    except:
        st.error("LLM output was not valid JSON.")
        st.write(response.content)
