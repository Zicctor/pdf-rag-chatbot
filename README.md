# 🤖 PDF RAG Chatbot with LangGraph and Together AI

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit for the UI, LangGraph for building the conversational agent, Together AI for the Large Language Model (LLM), ChromaDB for the vector store, and HuggingFace sentence-transformers for embeddings.

The chatbot allows you to chat with your PDF documents 📄. It automatically loads PDFs from a `documents` folder, creates or updates a vector database, and uses this database to answer your questions based on the document content.

## ✨ Features

*   **🖥️ Streamlit Interface:** Simple and interactive web UI for chatting.
*   **🧠 LangGraph Agent:** Manages the flow of conversation, including deciding when to retrieve information or generate a response.
*   **🔗 Together AI Integration:** Leverages powerful LLMs via the Together AI API.
*   **💾 ChromaDB Vector Store:** Stores and retrieves document embeddings efficiently.
*   **🔄 Automatic Synchronization:** Detects added or removed PDFs in the `documents` folder on startup and updates the vector store accordingly.
*   **📊 Sidebar Status:** Loading and synchronization messages are displayed neatly in the sidebar.

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Chatbot_Rag
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Configuration

1.  **🔑 Together AI API Key:** You need an API key from [Together AI](https://api.together.xyz/). The application will prompt you to enter this key in the Streamlit sidebar.
2.  **📄 PDF Documents:** Place the PDF files you want to chat with into the `Chatbot_Rag/documents` folder. Create this folder if it doesn't exist.

## ▶️ Running the Application

1.  Ensure you have placed your PDF files inside the `Chatbot_Rag/documents` directory.
2.  Navigate to the `Chatbot_Rag` directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your web browser. Enter your Together AI API Key in the sidebar.
5.  The app will load embeddings, create/load/synchronize the vector database (status shown in the sidebar), and then you can start asking questions about your documents.
