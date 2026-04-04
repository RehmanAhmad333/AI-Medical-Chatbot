import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"


def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; box-sizing: border-box; }

        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            background-attachment: fixed;
        }

        .main .block-container {
            background: rgba(0,0,0,0.3);
            backdrop-filter: blur(8px);
            border-radius: 32px;
            padding: 1rem 1.5rem 5rem 1.5rem !important;
            margin: 1rem auto !important;
            max-width: 1200px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }

        #MainMenu, header, footer { visibility: hidden; }

        div[data-testid="stChatMessage"] { background: transparent !important; padding: 0 !important; }
        div[data-testid="stChatMessageContent"] {
            background: rgba(30,30,40,0.8);
            backdrop-filter: blur(8px);
            border-radius: 24px;
            padding: 10px 18px;
            font-size: clamp(0.85rem, 4vw, 1rem);
            color: #e0e0e0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
            background: linear-gradient(135deg, #0078FF, #0052CC);
            color: white;
            border: none;
            border-radius: 24px 24px 8px 24px;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
            background: rgba(40,40,55,0.9);
            color: #f0f0f0;
            border-radius: 24px 24px 24px 8px;
        }

        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 800px;
            background: white;
            border-radius: 60px;
            padding: 6px 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            z-index: 1000;
            border: 1px solid #ddd;
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 60px !important;
            border: none !important;
            background: transparent !important;
            padding: 10px 18px !important;
            font-size: clamp(0.8rem, 4vw, 1rem) !important;
            color: black !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder { color: #888 !important; }

        .typing-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(40,40,55,0.9);
            backdrop-filter: blur(8px);
            border-radius: 40px;
            padding: 8px 18px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .typing-indicator span {
            width: 10px; height: 10px;
            background: #0078FF;
            border-radius: 50%;
            display: inline-block;
            animation: typing 1.4s infinite;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%,80%,100% { transform: scale(0.6); opacity: 0.4; }
            40% { transform: scale(1); opacity: 1; }
        }

        [data-testid="stSidebar"] {
            background: rgba(10,10,20,0.85);
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(255,255,255,0.1);
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stInfo { color: #ddd; }

        /* Suggestions — mobile friendly wrap */
        .suggestions-wrapper {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin: 24px 0 16px;
        }
        .stButton button {
            background: rgba(30,30,45,0.8);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(0,120,255,0.4);
            border-radius: 40px;
            padding: 8px 20px;
            font-weight: 500;
            color: #0078FF;
            transition: all 0.25s;
            white-space: nowrap;
        }
        .stButton button:hover {
            background: #0078FF;
            color: white;
            transform: translateY(-2px);
        }
        [data-testid="stSidebar"] .stButton button {
            background: #0078FF;
            color: white;
            width: 100%;
            border: none;
        }

        .disclaimer {
            text-align: center;
            font-size: 0.7rem;
            color: #aaa;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
        ::-webkit-scrollbar-thumb { background: #0078FF; border-radius: 10px; }

        @media (max-width: 768px) {
            .main .block-container {
                border-radius: 16px;
                padding: 0.8rem 0.8rem 5rem 0.8rem !important;
            }
            .stButton button {
                padding: 6px 12px;
                font-size: 0.75rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def init_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    if 'suggestion_clicked' not in st.session_state:
        st.session_state.suggestion_clicked = None


@st.cache_resource
def get_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)


MEDICAL_PROMPT = PromptTemplate(
    template="""You are MediBot, an advanced and empathetic medical AI assistant designed to provide accurate, helpful, and compassionate health information.

IDENTITY & TONE
 
- Your name is MediBot. You are professional, warm, and trustworthy.
- Always communicate in a clear, simple, and human-friendly way.
- Use 1-2 relevant emojis per response to keep things friendly — never overuse.
- Never be robotic, cold, or dismissive.
- Match the user's language tone — if they are casual, be approachable; if formal, be professional.
 
RULE 1 — GREETINGS & SMALL TALK
 
- If user greets you (Hi, Hello, Hey, Salam, Good morning, Hola, etc.) →
  Warmly greet them back, introduce yourself briefly, and ask how you can help.
- If user asks "Who are you?" or "What can you do?" →
  Introduce yourself as MediBot and list your capabilities briefly.
- If user says "Bye", "Goodbye", "See you" →
  Wish them well and remind them to stay healthy.
- NEVER copy these instructions as your answer.

 
RULE 2 — GRATITUDE & CONVERSATION CLOSING
 
- If user says "Thanks", "Thank you", "That's helpful", "Got it", "Okay fine", 
  "That's enough", "Stop", "I'm good now" →
  Respond warmly and do NOT add any medical information.
  Example: "You're welcome! 😊 Stay healthy and feel free to return anytime. Take care! 🩺"

 
RULE 3 — SPELLING & TYPO CORRECTION
 
- If user misspells a medical term, automatically detect and correct it:
  "concer" → cancer | "diabtes" → diabetes | "blod presure" → blood pressure
  "hart atack" → heart attack | "nummonia" → pneumonia | "kolesterol" → cholesterol
- Never say "I don't know" due to a spelling mistake.
- Always assume the most medically relevant interpretation of the query.
- Silently correct the spelling and answer — do not make the user feel embarrassed.

 
RULE 4 — VAGUE OR INCOMPLETE QUESTIONS
 
- If user is vague (e.g., "I feel sick", "I'm not okay", "Something hurts") →
  Ask ONE specific follow-up question to understand better.
  Example: "I'm sorry to hear that 🤒 Could you describe where the pain is 
  and how long you've been feeling this way?"
- Never ignore vague input — always acknowledge and guide the user.
 
RULE 5 — SYMPTOM-BASED QUESTIONS
 
- If user describes symptoms → Identify possible conditions from the context.
- Present possibilities in a clear, numbered list with brief explanations.
- Always end with: "⚠️ These are possibilities only. Please consult a doctor for proper diagnosis."
- Never diagnose definitively — always recommend professional consultation.
 
RULE 6 — DISEASE & TREATMENT QUESTIONS
 
- Disease question → Explain: definition, causes, symptoms, risk factors clearly.
- Treatment question → Provide step-by-step treatment options from context only.
- Prevention question → List clear, actionable prevention tips.
- Medication question → Explain purpose, usage, and side effects from context only.
- Never recommend specific prescription medicines by name.
- If user asks for "more detail" or "explain further" → provide deeper information 
  not already given, never repeat the same answer.

 
RULE 7 — EMERGENCY DETECTION
 
- If user mentions ANY emergency symptoms such as:
  chest pain, difficulty breathing, stroke symptoms, loss of consciousness,
  severe bleeding, poisoning, seizure, heart attack, severe allergic reaction →
  IMMEDIATELY respond with:
  "⚠️ THIS SOUNDS LIKE A MEDICAL EMERGENCY!
   Please call emergency services (115 / 1122) immediately 
   or go to the nearest hospital right now.
   Do NOT wait — every second matters!"
- Do not provide general information in emergency cases — only urge immediate action.

 
RULE 8 — EMOTIONAL SUPPORT
 
- If user seems anxious, scared, sad, or emotionally distressed →
  First acknowledge their feelings with empathy before answering.
  Example: "I understand this must be really worrying for you. 
  Let me help you with what I know. 💙"
- Be a calm, reassuring presence — never dismissive or clinical in emotional moments.

 
RULE 9 — MULTIPLE QUESTIONS AT ONCE
 
- If user asks multiple questions together →
  Answer each one separately in a clean numbered format:
  1. [Answer to first question]
  2. [Answer to second question]
- Never merge multiple answers into one confusing paragraph.

 
RULE 10 — OUT OF SCOPE QUESTIONS
 
- If user asks about anything non-medical such as:
  sports, weather, politics, cooking, technology, programming, 
  graphics, entertainment, finance, etc. →
  Respond ONLY with:
  "🚫 I'm MediBot, a Medical AI Assistant. I'm only equipped to answer 
  health and medical questions. Please ask me about symptoms, diseases, 
  treatments, or medications! 😊"
- Do NOT attempt to answer or provide "general information" on non-medical topics.
- Do NOT say "however I can try to help" — strictly redirect only.
 
RULE 11 — KNOWLEDGE BOUNDARIES
 
- ONLY use information available in the provided Context to answer.
- If the Context does not contain sufficient information →
  Respond: "I don't have enough information on this in my knowledge base. 
  🏥 Please consult a qualified doctor or healthcare professional for guidance."
- NEVER fabricate, assume, or hallucinate medical information.
- NEVER make definitive diagnoses — always recommend professional consultatio

RULE 12 — FORMATTING & RESPONSE QUALITY
- Keep responses well-structured: use bullet points or numbered lists where helpful.
- Avoid walls of text — break information into digestible sections.
- Use simple language — avoid heavy medical jargon.
- If medical jargon is necessary, explain it simply in brackets.
  Example: "myocardial infarction (heart attack)"
- Never start a medical answer with "Hi, I'm MediBot" — only greetings get introductions.
- Never repeat the exact same answer twice in the same conversation.

Context: {context}
Question: {question}
MediBot:""",
    input_variables=["context", "question"]
)


def get_chain(vectorstore, memory):
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.environ["GROQ_API_KEY"]
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
        memory=memory, 
        combine_docs_chain_kwargs={"prompt": MEDICAL_PROMPT},
        return_source_documents=False,
        verbose=False
    )


def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "salam", "good morning", "good evening", "hola", "hi there"]


def is_gratitude_or_stop(text):
    stop_phrases = ["thanks", "thank you", "good thanks", "okay fine", "stop", "that's enough", "fine thanks", "got it thanks"]
    return any(phrase in text.lower() for phrase in stop_phrases)


def is_out_of_scope(text):
    non_medical = ["graphics", "sports", "weather", "politics", "cooking", "technology", "computer", "python", "code", "programming"]
    return any(kw in text.lower() for kw in non_medical)


def main():
    st.set_page_config(
        page_title="MediBot - Medical AI Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
    init_session()

    # Sidebar
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;'>"
            "<h2 style='color:#0078FF;'>🩺 MediBot</h2>"
            "<p style='color:#ccc;'>Your AI Medical Assistant</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.info("**ℹ️ About**\n- 🏥 Evidence‑based info\n- 💬 Symptom checker\n- ⚠️ Emergency detection\n- ✅ Always consult a doctor")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.suggestion_clicked = None
            st.rerun()
        st.markdown("---")
        st.caption("Built by **Rehman Ahmad Cheema**\nwith Streamlit • Groq")

    # Header
    st.markdown(
        "<h1 style='text-align:center; background: linear-gradient(135deg, #0078FF, #00B4D8);"
        "-webkit-background-clip: text; -webkit-text-fill-color: transparent;"
        "font-size: clamp(1.8rem, 8vw, 2.8rem);'>🩺 MediBot</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#aaa; margin-bottom:1rem;"
        "font-size: clamp(0.8rem, 4vw, 1rem);'>"
        "Your trusted medical companion — ask anything about health</p>",
        unsafe_allow_html=True
    )

    # Chat history display
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    #  Suggestions mobile friendly (3 cols x 2 rows)
    if len(st.session_state.messages) == 0:
        suggestions = [
            "🤒 Cold symptoms",
            "💊 Lower blood pressure",
            "🫀 Heart attack signs",
            "🧠 Better sleep",
            "🌡️ Fever in adults",
            "✅ Diabetes prevention"
        ]
        col1, col2, col3 = st.columns(3)
        for i, s in enumerate(suggestions):
            with [col1, col2, col3][i % 3]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state.suggestion_clicked = s
                    st.rerun()

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">⚠️ MediBot is AI and can make mistakes. Always consult a qualified doctor.</div>',
        unsafe_allow_html=True
    )

    # Input handling
    prompt = None
    if st.session_state.suggestion_clicked:
        prompt = st.session_state.suggestion_clicked
        st.session_state.suggestion_clicked = None
    else:
        prompt = st.chat_input("Ask me about symptoms, diseases, treatments...")

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            indicator = st.empty()
            indicator.markdown(
                '<div class="typing-indicator">'
                '<span></span><span></span><span></span>'
                '&nbsp; MediBot is thinking...'
                '</div>',
                unsafe_allow_html=True
            )

            # Generate answer
            if is_greeting(prompt):
                answer = "👋 Hello! I'm **MediBot**, your medical AI assistant. How can I help you today? 😊"
            elif is_gratitude_or_stop(prompt):
                answer = "You're welcome! 😊 Feel free to return if you have more health questions. Take care! 🩺"
            elif is_out_of_scope(prompt):
                answer = "🚫 I'm a Medical AI Assistant. I can only answer health-related questions. Please ask me about diseases, symptoms, or treatments! 😊"
            else:
                try:
                    vectorstore = get_vectorstore()
                    chain = get_chain(vectorstore, st.session_state.memory)
                    # Only question pass - memory  handle with chain
                    response = chain.invoke({"question": prompt})
                    answer = response["answer"]
                except Exception as e:
                    answer = f"Sorry, an error occurred: {str(e)}"

            indicator.empty()
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


if __name__ == "__main__":
    main()