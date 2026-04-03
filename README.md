<!DOCTYPE html>
<html>
<body>
<div style="font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); color: #eef2ff; border-radius: 24px;">

<!-- HEADER SECTION -->
<div style="text-align: center; padding: 2rem 1rem 1rem 1rem;">
    <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #0078FF, #00B4D8); -webkit-background-clip: text; background-clip: text; color: transparent; margin: 0;">🩺 MediBot</h1>
    <p style="font-size: 1.2rem; color: #a0a8c0; margin-top: 0.5rem;">Your AI‑powered Medical Assistant – RAG + LLM</p>
    <div style="margin-top: 1.5rem;">
        <span style="background: #2a2f45; padding: 6px 14px; border-radius: 40px; font-size: 0.8rem;">🎯 LangChain</span>
        <span style="background: #2a2f45; padding: 6px 14px; border-radius: 40px; font-size: 0.8rem;">📊 FAISS</span>
        <span style="background: #2a2f45; padding: 6px 14px; border-radius: 40px; font-size: 0.8rem;">🤖 Groq LLM</span>
        <span style="background: #2a2f45; padding: 6px 14px; border-radius: 40px; font-size: 0.8rem;">🧠 HuggingFace</span>
        <span style="background: #2a2f45; padding: 6px 14px; border-radius: 40px; font-size: 0.8rem;">🎨 Streamlit</span>
    </div>
</div>

<hr style="border-color: #2a2f45; margin: 20px 0;">

<!-- OVERVIEW -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">📌 Overview</h2>
    <p><strong>MediBot</strong> is a <strong>Retrieval‑Augmented Generation (RAG)</strong> medical chatbot. It ingests medical PDFs (e.g., <em>The Gale Encyclopedia of Medicine</em>), builds a FAISS vector store, and uses a Groq LLM (Llama‑3.1‑8B) or HuggingFace model to answer health questions. The app features a modern, responsive chat interface with memory, emergency detection, and out‑of‑scope handling.</p>
</div>

<!-- FEATURES -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">✨ Key Features</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        <div style="background: #0b0f1c; padding: 1rem; border-radius: 16px;"><strong>📚 RAG Pipeline</strong><br>Load PDF → chunk → embed → FAISS → retrieve → LLM generation</div>
        <div style="background: #0b0f1c; padding: 1rem; border-radius: 16px;"><strong>🧠 Conversational Memory</strong><br>Remembers last 5 exchanges (ConversationBufferWindowMemory)</div>
        <div style="background: #0b0f1c; padding: 1rem; border-radius: 16px;"><strong>⚡ Fast Inference</strong><br>Groq Llama‑3.1‑8B (or Mistral via HuggingFace)</div>
        <div style="background: #0b0f1c; padding: 1rem; border-radius: 16px;"><strong>🖥️ Modern UI</strong><br>Dark glass‑morphism, responsive, typing indicator, suggestion pills</div>
        <div style="background: #0b0f1c; padding: 1rem; border-radius: 16px;"><strong>🚨 Emergency Detection</strong><br>Immediate warning for critical symptoms</div>
        <div style="background: #0b0f1c; padding: 1rem; border-radius: 16px;"><strong>📱 Mobile Ready</strong><br>Hamburger sidebar, adaptive font sizes, touch‑friendly</div>
    </div>
</div>

<!-- TECH STACK -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">🛠️ Tech Stack</h2>
    <ul style="columns: 2; column-gap: 2rem; list-style: none; padding-left: 0;">
        <li>🐍 <strong>Python 3.11+</strong></li>
        <li>📦 <strong>LangChain</strong> – orchestration</li>
        <li>🔍 <strong>FAISS</strong> – vector database</li>
        <li>🤗 <strong>HuggingFace Embeddings</strong> – sentence-transformers/all-MiniLM-L6-v2</li>
        <li>⚡ <strong>Groq API</strong> – Llama‑3.1‑8B (or Mistral via HF)</li>
        <li>🎨 <strong>Streamlit</strong> – frontend</li>
        <li>📄 <strong>PyPDF / DirectoryLoader</strong> – PDF ingestion</li>
        <li>📝 <strong>RecursiveCharacterTextSplitter</strong> – chunking</li>
    </ul>
</div>

<!-- PROJECT STRUCTURE -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">📁 Project Structure</h2>
    <pre style="background: #0b0f1c; padding: 1rem; border-radius: 16px; overflow-x: auto; color: #b0c4de;">
AI_MEDICAL_CHATBOT/
├── data/
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE.pdf
├── vectorstore/
│   └── db_faiss/
│       ├── index.faiss
│       └── index.pkl
├── venv/                     # virtual environment (ignored)
├── .env                      # GROQ_API_KEY / HF_TOKEN
├── .gitignore
├── setup_memory_for_llm.py   # creates vector store from PDF
├── connect_memory_with_llm.py# test RAG chain
├── medibot_ui.py             # main Streamlit app
└── requirements.txt
    </pre>
</div>

<!-- SETUP & INSTALLATION -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">🚀 Setup & Installation</h2>
    <ol style="line-height: 1.7;">
        <li><strong>Clone the repository</strong><br>
            <code>git clone https://github.com/RehmanAhmad333/AI-Medical-Chatbot.git</code><br>
            <code>cd AI-Medical-Chatbot</code>
        </li>
        <li><strong>Create virtual environment</strong><br>
            <code>python -m venv venv</code><br>
            <code>venv\Scripts\activate</code> (Windows) / <code>source venv/bin/activate</code> (Mac/Linux)
        </li>
        <li><strong>Install dependencies</strong><br>
            <code>pip install -r requirements.txt</code>
        </li>
        <li><strong>Set up environment variables</strong><br>
            Create <code>.env</code> file:<br>
            <code>GROQ_API_KEY=your_groq_api_key</code><br>
            <code>HF_TOKEN=your_huggingface_token</code> (if using HuggingFace models)
        </li>
        <li><strong>Build FAISS vector store (first time only)</strong><br>
            <code>python setup_memory_for_llm.py</code><br>
            This will process the PDF and create the <code>vectorstore/db_faiss</code> folder.
        </li>
        <li><strong>Run the Streamlit app</strong><br>
            <code>streamlit run medibot_ui.py</code>
        </li>
    </ol>
</div>

<!-- USAGE -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">💬 Usage</h2>
    <ul>
        <li>Type a medical question (e.g., <em>"What are the symptoms of diabetes?"</em>)</li>
        <li>Click suggestion pills for quick queries</li>
        <li>The bot will answer using the medical PDF context</li>
        <li>Ask follow‑up questions – memory keeps context</li>
        <li>Emergency keywords trigger immediate safety warnings</li>
        <li>Non‑medical questions are politely rejected</li>
        <li>Use <strong>Clear Chat</strong> button to reset conversation</li>
    </ul>
    <img src="image.png" alt="Screenshot placeholder" style="max-width:100%; border-radius:16px; margin-top:10px;">
</div>

<!-- FUTURE ENHANCEMENTS -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">🔮 Future Enhancements</h2>
    <ul>
        <li>✨ Support for multiple medical PDFs (upload + reindex)</li>
        <li>✨ User login & chat history storage (SQLite)</li>
        <li>✨ Voice input / text‑to‑speech</li>
        <li>✨ Confidence scores & source citations</li>
        <li>✨ Docker container for one‑click deployment</li>
        <li>✨ Fine‑tune LLM on medical QA dataset</li>
    </ul>
</div>

<!-- CONTRIBUTING -->
<div style="background: #1e2338; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <h2 style="color: #00B4D8; margin-top: 0;">🤝 Contributing</h2>
    <p>Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.</p>
</div>

<!-- LICENSE & AUTHOR -->
<div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px; background: #0b0f1c; border-radius: 24px; padding: 1.5rem; margin: 30px 0;">
    <div>
        <h3 style="color: #00B4D8; margin: 0 0 8px 0;">📄 License</h3>
        <p>MIT © 2025 Rehman Ahmad Cheema</p>
    </div>
    <div>
        <h3 style="color: #00B4D8; margin: 0 0 8px 0;">👤 Author</h3>
        <p><strong>Rehman Ahmad Cheema</strong><br>
        🔗 <a href="https://github.com/RehmanAhmad333" style="color: #00B4D8;">GitHub</a> • 
        💼 <a href="https://www.linkedin.com/in/rehman-ahmad-9a5b17384/" style="color: #00B4D8;">LinkedIn</a>
        </p>
    </div>
</div>

<!-- FOOTER -->
<div style="text-align: center; padding: 1rem 0; border-top: 1px solid #2a2f45; margin-top: 20px; color: #6f7a9e;">
    <p>Built with ❤️ by Rehman Ahmad Cheema | Streamlit + LangChain + Groq</p>
</div>

</div>
</body>
</html>