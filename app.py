import os
import json
import re
import tempfile
from textwrap import dedent
from html import escape

import pandas as pd
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
# CONFIG
# ==============================
st.set_page_config(page_title="Back Office AI Router", layout="wide")

# ==============================
# STYLING
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    color: #0f172a;
}
.panel {
    background: white;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.result-card {
    background: white;
    border-radius: 16px;
    padding: 16px;
    margin-top: 10px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(15,23,42,0.06);
}
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}
.small-note {
    color: #475569;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# ENV
# ==============================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Put it in your .env file first.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


# ==============================
# ROUTING LOGIC
# ==============================
ROUTING_MAP = {
    "Invoice": "Finance / Accounts Payable",
    "Purchase Order": "Procurement / Supply Chain",
    "Contract": "Legal / Compliance",
    "HR Document": "Human Resources",
    "Internal Memo": "Operations / Admin",
    "Financial Report": "Finance / Management"
}

def suggest_routing(doc_type: str) -> str:
    return ROUTING_MAP.get(doc_type, "Back Office Review")


# ==============================
# JSON CLEANER
# ==============================
def clean_llm_json(text: str) -> str:
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()


# ==============================
# LLM CHAIN
# ==============================
@st.cache_resource
def build_chain():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are an office document classification assistant.

Classify into one of:
Invoice, Purchase Order, Contract, HR Document, Internal Memo, Financial Report.

Return ONLY valid JSON. Do not add markdown, code fences, or extra text.

Format:
{{
  "document_type": "...",
  "confidence": 0.0,
  "recommended_department": "...",
  "reasoning": "..."
}}

Document:
{document_text}
""")
    return prompt | llm

chain = build_chain()


# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="panel">
    <h1>Back Office AI Router</h1>
    <p>Upload → Classify → Route documents automatically</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel">
    <b>System Features:</b>
    <ul>
        <li>AI-based classification</li>
        <li>Confidence scoring</li>
        <li>Routing suggestions</li>
        <li>Back-office automation</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ==============================
# UPLOAD
# ==============================
st.markdown("### Upload documents")
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully")


# ==============================
# RESULTS STORAGE
# ==============================
results_data = []


# ==============================
# PROCESS
# ==============================
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        st.subheader(file_name)

        temp_path = None
        try:
            file_bytes = uploaded_file.getvalue()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            # Loader
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_path)
            elif file_extension == "docx":
                loader = Docx2txtLoader(temp_path)
            elif file_extension == "txt":
                loader = TextLoader(temp_path, encoding="utf-8")
            else:
                st.error("Unsupported file")
                continue

            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            text = "\n\n".join(
                [d.page_content for d in splitter.split_documents(docs)]
            )[:6000]

            with st.spinner("Analyzing document..."):
                response = chain.invoke({"document_text": text})

            cleaned = clean_llm_json(response.content)
            result = json.loads(cleaned)

            doc_type = str(result.get("document_type", "Unknown"))
            confidence = float(result.get("confidence", 0))
            department = str(result.get("recommended_department", "Unknown"))
            reasoning = str(result.get("reasoning", "No reasoning provided."))

            routing = suggest_routing(doc_type)

            # Confidence label
            if confidence >= 0.85:
                label = "High"
                color = "#dcfce7"
                label_text = "#166534"
            elif confidence >= 0.60:
                label = "Medium"
                color = "#fef3c7"
                label_text = "#92400e"
            else:
                label = "Low"
                color = "#fee2e2"
                label_text = "#991b1b"

            # Clean display in normal English
            st.markdown(
                dedent(f"""
                <div class="result-card">
                    <h3>{escape(doc_type)}</h3>
                    <p><b>Department:</b> {escape(department)}</p>
                    <p><b>Routing Suggestion:</b> {escape(routing)}</p>
                    <p><b>Reason:</b> {escape(reasoning)}</p>
                </div>
                """),
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="badge" style="background:{color}; color:{label_text};">
                    Confidence: {confidence:.2f} ({label})
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)

            results_data.append({
                "File Name": file_name,
                "Document Type": doc_type,
                "Department": department,
                "Routing Suggestion": routing,
                "Confidence": round(confidence, 2),
                "Reason": reasoning
            })

        except Exception as e:
            st.error("Could not parse AI response")
            st.caption(str(e))
            st.write(response.content if 'response' in locals() else "")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    # ==============================
    # DOWNLOAD CSV
    # ==============================
    if results_data:
        st.markdown("### Export results")
        df = pd.DataFrame(results_data)
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="document_classification_results.csv",
            mime="text/csv"
        )


# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Back Office AI Router • AI-powered document workflow system")