# 🛡️ AI Text Summarizer

A "Zero-Hallucination" summarizer and Question-Answering application powered by **Google Gemini Pro Latest** and **LangChain** (RAG Repository). 

This tool is designed for strict document analysis. Unlike standard AI chatbots, it refuses to answer questions if the information is not explicitly found within the uploaded PDF or DOCX file.

## 🚀 Features

-   **Strict Context Mode:** The AI acts as a precise analyst and will not use outside knowledge.
-   **Multi-Format Support:** Upload PDF or DOCX files.
-   **Secure Configuration:** API keys are managed securely via environment variables.
-   **Smart Summary:** Generates factual executive summaries using Map-Reduce chains.
-   **Context Detection:** Automatically detects if the document is a legal contract, technical manual, etc., and adjusts its persona.

## 🛠️ Tech Stack

-   **Frontend:** Streamlit
-   **LLM:** Google Gemini Pro Latest
-   **Orchestration:** LangChain
-   **Vector Store:** FAISS (Local CPU indexing)
-   **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/myprofilehub/AI-Text-Summarizer.git](https://github.com/myprofilehub/AI-Text-Summarizer.git)
    cd AI-Text-Summarizer
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    Create a file named `.env` in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run textsummarizer.py
