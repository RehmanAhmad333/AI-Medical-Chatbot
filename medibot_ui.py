# # import os
# # import streamlit as st

# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_core.prompts import PromptTemplate
# # from langchain_groq import ChatGroq
# # from langchain.memory import ConversationBufferWindowMemory
# # from langchain.chains import ConversationalRetrievalChain

# # from dotenv import load_dotenv, find_dotenv
# # load_dotenv(find_dotenv())

# # DB_FAISS_PATH = "vectorstore/db_faiss"

# # @st.cache_resource
# # def get_vectorstore(db_faiss_path):
# #     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #     db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
# #     return db

# # def set_custom_prompt(custom_prompt_template):
# #     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
# #     return prompt

# # CUSTOM_PROMPT_TEMPLATE = """
# # You are MediBot, a knowledgeable, empathetic, and careful medical AI assistant.

# # SECTION 1 - GREETING & SMALL TALK
# # - If user sends a greeting (Hi, Hello, Hey, Salam, etc.) →
# #   Warmly introduce yourself as MediBot and ask how you can help.
# # - Do NOT copy this instruction as your answer.
# # - If user says Thank you or Bye → respond politely.

# # SECTION 2 - SPELLING & TYPO HANDLING
# # - If the user misspells a medical term (e.g., "concer" → cancer, "diabtes" → diabetes,
# #   "blod presure" → blood pressure, "hart atack" → heart attack) →
# #   Automatically understand the intended word and answer correctly.
# # - Never say "I don't know" just because of a spelling mistake.
# # - Always assume the most medically relevant interpretation.

# # SECTION 3 - VAGUE OR INCOMPLETE QUESTIONS
# # - If the user is vague (e.g., "I feel sick", "I'm not well", "I have pain") →
# #   Ask ONE specific follow-up question to clarify:
# #   "Could you describe your symptoms in more detail? 
# #   For example: where is the pain, how long have you had it?"
# # - Never ignore a vague question — always try to help.

# # SECTION 4 - SYMPTOM-BASED QUESTIONS
# # - If user describes symptoms → Match them to possible conditions from the context.
# # - List possible conditions clearly with brief explanation.
# # - Always add: "Please consult a doctor for proper diagnosis."

# # SECTION 5 - DISEASE / TREATMENT QUESTIONS
# # - Disease question → Explain: definition, causes, symptoms, risk factors.
# # - Treatment question → Give step-by-step treatment from context only.
# # - Prevention question → Give clear prevention tips from context.
# # - Medication question → Explain usage and side effects from context only.

# # SECTION 6 - EMERGENCY DETECTION
# # - If user mentions emergency symptoms like:
# #   chest pain, can't breathe, heart attack, stroke, unconscious, 
# #   severe bleeding, poisoning, seizure → 
# #   IMMEDIATELY respond:
# #   "⚠️ THIS SOUNDS LIKE A MEDICAL EMERGENCY!
# #   Please call emergency services (115/1122) immediately or go to the nearest hospital.
# #   Do not wait — this requires urgent medical attention!"

# # SECTION 7 - EMOTIONAL DISTRESS
# # - If user seems worried, scared, or anxious about their health →
# #   First acknowledge their feelings:
# #   "I understand this can be worrying. Let me help you with the information I have."
# #   Then answer their question calmly and clearly.

# # SECTION 8 - OUT OF SCOPE QUESTIONS
# # - If the user asks anything that is NOT related to medicine, health, 
# #   diseases, symptoms, or treatments → you MUST respond with ONLY this:
# #   "🚫 I'm MediBot, a Medical AI Assistant. I can only answer 
# #   health-related questions. Please ask me about diseases, 
# #   symptoms, or treatments! 😊"
# # - Do NOT attempt to answer non-medical questions AT ALL.
# # - Do NOT say "however I can try..." — strictly refuse and redirect.
# # - Non-medical topics include: graphics, sports, weather, 
# #   cooking, technology, politics, etc.

# # SECTION 9 - MULTIPLE QUESTIONS AT ONCE
# # - If user asks multiple questions together →
# #   Answer each one clearly in numbered format:
# #   1. First answer...
# #   2. Second answer...

# # SECTION 10 - GENERAL RULES
# # - NEVER make up medical information not present in context.
# # - NEVER recommend specific prescription medicines by name.
# # - ALWAYS suggest consulting a real doctor for serious conditions.
# # - Keep answers clear, structured, and easy to understand.
# # - Use simple language — avoid heavy medical jargon unless necessary.
# # - If jargon is used, explain it simply in brackets.
# # - ALWAYS use relevant emojis in your responses to make them friendly and engaging.
# #   For example:
# #   🏥 for hospital/medical topics
# #   💊 for medicines
# #   🤒 for symptoms/illness
# #   ❤️ for heart related
# #   🧠 for brain related
# #   ⚠️ for warnings
# #   ✅ for tips/recommendations
# #   😊 for greetings
# #   🌡️ for fever/temperature
# # - Do not overuse emojis — use 1-2 per point maximum.

# # Context:
# # {context}

# # Question:
# # {question}

# # Answer:
# # """

# # def main():
# #     st.title("Your Medical AI Assistant")

# #     # Messages initialize karo
# #     if 'message' not in st.session_state:
# #         st.session_state.message = []

# #     # Memory session include not reset 
# #     if 'memory' not in st.session_state:
# #         st.session_state.memory = ConversationBufferWindowMemory(
# #             k=3,
# #             memory_key="chat_history",
# #             return_messages=True,
# #             output_key="answer"
# #         )

# #     for message in st.session_state.message:
# #         st.chat_message(message["role"]).markdown(message["content"])

# #     prompt = st.chat_input("Ask your medical question here...")

# #     if prompt:
# #         st.chat_message("user").markdown(prompt)
# #         st.session_state.message.append({"role": "user", "content": prompt})

# #         try:
# #             vectorstore = get_vectorstore(DB_FAISS_PATH)
# #             if vectorstore is None:
# #                 st.error("Failed to load vector store. Please check the path and try again.")
# #                 return

# #             qa_chain = ConversationalRetrievalChain.from_llm(
# #                 llm=ChatGroq(
# #                     model_name="llama-3.1-8b-instant",
# #                     temperature=0.0,
# #                     groq_api_key=os.environ["GROQ_API_KEY"],
# #                 ),
# #                 retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
# #                 memory=st.session_state.memory,   # session memory use 
# #                 return_source_documents=True,
# #                 combine_docs_chain_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# #             )

# #             response = qa_chain.invoke({"question": prompt})
# #             result = response['answer']

# #             st.chat_message("assistant").markdown(result)
# #             st.session_state.message.append({"role": "assistant", "content": result})

# #         except Exception as e:
# #             st.error(f"An error occurred: {e}")

# # if __name__ == "__main__":
# #     main()


# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# DB_FAISS_PATH = "vectorstore/db_faiss"

# # ========== CUSTOM CSS (modern, responsive, medical theme) ==========
# def inject_custom_css():
#     st.markdown("""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
#         * { font-family: 'Inter', sans-serif; }
#         .stApp { background: #f4f7fc; }
#         /* Hide default Streamlit elements */
#         #MainMenu, header, footer { visibility: hidden; }
#         .main .block-container { padding-top: 1rem; padding-bottom: 5rem; max-width: 1100px; }
        
#         /* Chat bubbles */
#         div[data-testid="stChatMessage"] {
#             background: transparent !important;
#             box-shadow: none !important;
#             padding: 0 !important;
#         }
#         div[data-testid="stChatMessageContent"] {
#             background: white;
#             border-radius: 20px;
#             padding: 12px 18px;
#             box-shadow: 0 1px 2px rgba(0,0,0,0.05);
#         }
#         /* User message */
#         div[data-testid="stChatMessage"][data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
#             background: #0078FF;
#             color: white;
#             border-radius: 20px 20px 4px 20px;
#         }
#         /* Assistant message */
#         div[data-testid="stChatMessage"][data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
#             background: #e9ecef;
#             color: #1a1f36;
#             border-radius: 20px 20px 20px 4px;
#         }
#         /* Fixed chat input */
#         div[data-testid="stChatInput"] {
#             position: fixed;
#             bottom: 0;
#             left: 0;
#             right: 0;
#             background: white;
#             padding: 12px 20px;
#             border-top: 1px solid #dee2e6;
#             box-shadow: 0 -2px 10px rgba(0,0,0,0.02);
#             z-index: 1000;
#         }
#         /* Typing indicator */
#         .typing-indicator {
#             display: flex;
#             align-items: center;
#             gap: 6px;
#             background: #e9ecef;
#             border-radius: 20px;
#             padding: 10px 16px;
#             width: fit-content;
#         }
#         .typing-indicator span {
#             width: 8px;
#             height: 8px;
#             background: #6c757d;
#             border-radius: 50%;
#             animation: typing 1.4s infinite;
#         }
#         @keyframes typing { 0%,80%,100%{transform:scale(0.6);opacity:0.4;} 40%{transform:scale(1);opacity:1;} }
#         /* Sidebar */
#         [data-testid="stSidebar"] {
#             background: white;
#             border-right: 1px solid #e9ecef;
#         }
#         /* Suggestion pills */
#         .suggestion-pill {
#             background: white;
#             border: 1px solid #dee2e6;
#             border-radius: 40px;
#             padding: 6px 14px;
#             font-size: 0.8rem;
#             cursor: pointer;
#             transition: all 0.2s;
#         }
#         .suggestion-pill:hover {
#             background: #0078FF;
#             border-color: #0078FF;
#             color: white;
#         }
#         @media (max-width: 768px) {
#             .suggestion-pill { font-size: 0.7rem; padding: 4px 10px; }
#         }
#     </style>
#     """, unsafe_allow_html=True)

# # ========== SESSION STATE INIT ==========
# def init_session():
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'memory' not in st.session_state:
#         st.session_state.memory = ConversationBufferWindowMemory(
#             k=5, memory_key="chat_history", return_messages=True, output_key="answer"
#         )
#     if 'greeting_done' not in st.session_state:
#         st.session_state.greeting_done = False

# # ========== VECTOR STORE ==========
# @st.cache_resource
# def get_vectorstore():
#     embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)

# # ========== IMPROVED PROMPT (concise & forceful) ==========
# MEDICAL_PROMPT = PromptTemplate(
#     template="""You are MediBot, a medical AI assistant. Follow these rules strictly:

# 1. If the user greets (hi, hello, salam, hey): respond with a warm greeting, introduce yourself as MediBot, and ask how you can help. Do NOT use context.

# 2. If the user asks a non‑medical question (sports, weather, graphics, politics, etc.): reply: "🚫 I'm MediBot, a Medical AI Assistant. I can only answer health‑related questions. Please ask me about diseases, symptoms, or treatments! 😊"

# 3. For medical questions:
#    - Use the provided Context to answer accurately.
#    - If the answer is not in context, say "I don't have enough information. Please consult a doctor."
#    - For symptoms: list possible conditions and always add "Please consult a doctor for proper diagnosis."
#    - For emergencies (chest pain, breathing trouble, stroke, severe bleeding, seizure): respond with "⚠️ THIS SOUNDS LIKE A MEDICAL EMERGENCY! Call emergency services immediately."

# 4. Be empathetic, clear, and use 1-2 relevant emojis per answer.

# Context: {context}

# Chat History: {chat_history}

# User: {question}
# MediBot:""",
#     input_variables=["context", "chat_history", "question"]
# )

# # ========== CHAIN WITH CUSTOM PROMPT ==========
# def get_chain(vectorstore, memory):
#     llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=os.environ["GROQ_API_KEY"])
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": MEDICAL_PROMPT},
#         return_source_documents=False,
#         verbose=False
#     )

# # ========== HANDLE GREETINGS & SMALL TALK (bypass RAG) ==========
# def is_greeting(text):
#     greetings = ["hi", "hello", "hey", "salam", "good morning", "good evening", "yo", "hii", "helo"]
#     return text.lower().strip() in greetings or text.lower().startswith(("hi", "helo"))

# def is_out_of_scope(text):
#     non_medical = ["graphics", "sports", "weather", "politics", "cooking", "technology", "what is computer", "who are you"]
#     return any(word in text.lower() for word in non_medical)

# def get_greeting_response():
#     return "👋 Hello! I'm **MediBot**, your medical AI assistant. How can I help you today? 😊"

# def get_out_of_scope_response():
#     return "🚫 I'm MediBot, a Medical AI Assistant. I can only answer health‑related questions. Please ask me about diseases, symptoms, or treatments! 😊"

# # ========== MAIN APP ==========
# def main():
#     st.set_page_config(page_title="MediBot - Medical AI Assistant", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
#     inject_custom_css()
#     init_session()

#     # Sidebar
#     with st.sidebar:
#         st.markdown("<div style='text-align:center;'><h2>🩺 MediBot</h2><p style='color:#6c757d;'>Your AI Medical Assistant</p></div>", unsafe_allow_html=True)
#         st.markdown("---")
#         st.info("**ℹ️ About**\n- 🏥 Medical info from trusted sources\n- 💬 Symptom checker\n- ⚠️ Emergency detection\n- ✅ Always consult a doctor")
#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.messages = []
#             st.session_state.memory.clear()
#             st.session_state.greeting_done = False
#             st.rerun()
#         st.markdown("---")
#         st.caption("Built with Streamlit • Powered by Groq")

#     # Main chat area
#     st.markdown("<h1 style='text-align:center; background: linear-gradient(135deg, #0078FF, #00B4D8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>MediBot</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align:center; color:#6c757d;'>Your trusted medical companion — ask anything about health</p>", unsafe_allow_html=True)

#     # Display chat history
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
#             st.markdown(msg["content"])

#     # Suggestion pills (only when no messages)
#     if len(st.session_state.messages) == 0:
#         cols = st.columns(3)
#         suggestions = ["🤒 Cold symptoms", "💊 Lower blood pressure", "🫀 Heart attack signs", "🧠 Better sleep", "🌡️ Fever in adults", "✅ Diabetes prevention"]
#         for i, s in enumerate(suggestions):
#             with cols[i % 3]:
#                 if st.button(s, key=f"sug_{i}", use_container_width=True):
#                     st.session_state.suggestion_clicked = s
#                     st.rerun()

#     # Process user input
#     prompt = None
#     if 'suggestion_clicked' in st.session_state and st.session_state.suggestion_clicked:
#         prompt = st.session_state.suggestion_clicked
#         st.session_state.suggestion_clicked = None
#     else:
#         prompt = st.chat_input("Ask me about symptoms, diseases, treatments...")

#     if prompt:
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(prompt)

#         # Show typing indicator
#         with st.chat_message("assistant", avatar="🤖"):
#             indicator = st.empty()
#             indicator.markdown('<div class="typing-indicator"><span></span><span></span><span></span><span style="margin-left:8px;">MediBot is thinking...</span></div>', unsafe_allow_html=True)

#         # Route: greeting or out-of-scope -> direct answer
#         if is_greeting(prompt):
#             answer = get_greeting_response()
#             indicator.empty()
#             with st.chat_message("assistant", avatar="🤖"):
#                 st.markdown(answer)
#         elif is_out_of_scope(prompt):
#             answer = get_out_of_scope_response()
#             indicator.empty()
#             with st.chat_message("assistant", avatar="🤖"):
#                 st.markdown(answer)
#         else:
#             try:
#                 vectorstore = get_vectorstore()
#                 chain = get_chain(vectorstore, st.session_state.memory)
#                 # Note: we pass chat_history manually because the chain expects it in the prompt
#                 response = chain.invoke({"question": prompt, "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]})
#                 answer = response["answer"]
#                 indicator.empty()
#                 with st.chat_message("assistant", avatar="🤖"):
#                     st.markdown(answer)
#             except Exception as e:
#                 indicator.empty()
#                 answer = f"⚠️ Sorry, I encountered an error: {str(e)}"
#                 with st.chat_message("assistant", avatar="🤖"):
#                     st.markdown(answer)

#         st.session_state.messages.append({"role": "assistant", "content": answer})
#         st.rerun()

# if __name__ == "__main__":
#     main()



# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# DB_FAISS_PATH = "vectorstore/db_faiss"

# # ========== CUSTOM CSS (modern, responsive, centered pills) ==========
# def inject_custom_css():
#     st.markdown("""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
#         * { font-family: 'Inter', sans-serif; }
#         .stApp { background: #f4f7fc; }
#         #MainMenu, header, footer { visibility: hidden; }
#         .main .block-container { padding-top: 1rem; padding-bottom: 5rem; max-width: 1100px; margin: 0 auto; }
        
#         /* Chat bubbles */
#         div[data-testid="stChatMessage"] {
#             background: transparent !important;
#             box-shadow: none !important;
#             padding: 0 !important;
#         }
#         div[data-testid="stChatMessageContent"] {
#             background: white;
#             border-radius: 20px;
#             padding: 12px 18px;
#             box-shadow: 0 1px 2px rgba(0,0,0,0.05);
#         }
#         /* User message */
#         div[data-testid="stChatMessage"][data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
#             background: #0078FF;
#             color: white;
#             border-radius: 20px 20px 4px 20px;
#         }
#         /* Assistant message */
#         div[data-testid="stChatMessage"][data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
#             background: #e9ecef;
#             color: #1a1f36;
#             border-radius: 20px 20px 20px 4px;
#         }
#         /* Fixed chat input */
#         div[data-testid="stChatInput"] {
#             position: fixed;
#             bottom: 0;
#             left: 0;
#             right: 0;
#             background: white;
#             padding: 12px 20px;
#             border-top: 1px solid #dee2e6;
#             box-shadow: 0 -2px 10px rgba(0,0,0,0.02);
#             z-index: 1000;
#         }
#         /* Typing indicator */
#         .typing-indicator {
#             display: flex;
#             align-items: center;
#             gap: 6px;
#             background: #e9ecef;
#             border-radius: 20px;
#             padding: 10px 16px;
#             width: fit-content;
#         }
#         .typing-indicator span {
#             width: 8px;
#             height: 8px;
#             background: #6c757d;
#             border-radius: 50%;
#             animation: typing 1.4s infinite;
#         }
#         @keyframes typing { 0%,80%,100%{transform:scale(0.6);opacity:0.4;} 40%{transform:scale(1);opacity:1;} }
#         /* Sidebar */
#         [data-testid="stSidebar"] {
#             background: white;
#             border-right: 1px solid #e9ecef;
#         }
#         /* Suggestion pills - centered and responsive */
#         .suggestions-grid {
#             display: flex;
#             flex-wrap: wrap;
#             justify-content: center;
#             gap: 12px;
#             margin: 20px 0 10px;
#         }
#         .suggestion-pill {
#             background: white;
#             border: 1px solid #dee2e6;
#             border-radius: 40px;
#             padding: 8px 18px;
#             font-size: 0.85rem;
#             cursor: pointer;
#             transition: all 0.2s;
#             white-space: nowrap;
#         }
#         .suggestion-pill:hover {
#             background: #0078FF;
#             border-color: #0078FF;
#             color: white;
#             transform: translateY(-2px);
#         }
#         @media (max-width: 768px) {
#             .suggestion-pill { font-size: 0.7rem; padding: 6px 12px; white-space: normal; }
#             .suggestions-grid { gap: 8px; }
#         }
#     </style>
#     """, unsafe_allow_html=True)

# # ========== SESSION STATE INIT ==========
# def init_session():
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'memory' not in st.session_state:
#         st.session_state.memory = ConversationBufferWindowMemory(
#             k=5, memory_key="chat_history", return_messages=True, output_key="answer"
#         )

# # ========== VECTOR STORE ==========
# @st.cache_resource
# def get_vectorstore():
#     embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)

# # ========== IMPROVED PROMPT (no greeting for medical questions) ==========
# MEDICAL_PROMPT = PromptTemplate(
#     template="""You are MediBot, a strict medical AI assistant. Follow these rules:

# 1. If the user's message is ONLY a simple greeting like "hi", "hello", "hey", "salam" → respond with a warm greeting and ask how you can help.
# 2. If the user asks anything NOT related to medicine, health, symptoms, diseases, or treatments → respond: "🚫 I'm MediBot, a Medical AI Assistant. I can only answer health‑related questions. Please ask me about diseases, symptoms, or treatments! 😊"
# 3. For medical questions:
#    - Use the provided Context to answer accurately.
#    - If the answer is not in context, say "I don't have enough information. Please consult a doctor."
#    - For symptoms: list possible conditions and always add "Please consult a doctor for proper diagnosis."
#    - For emergencies (chest pain, breathing trouble, stroke, severe bleeding, seizure): respond with "⚠️ THIS SOUNDS LIKE A MEDICAL EMERGENCY! Call emergency services immediately."
# 4. Be empathetic, clear, and use 1-2 relevant emojis per answer.
# 5. NEVER start a medical answer with a greeting like "Hi, I'm MediBot". Answer directly.

# Context: {context}

# Chat History: {chat_history}

# User: {question}
# MediBot:""",
#     input_variables=["context", "chat_history", "question"]
# )

# # ========== CHAIN WITH CUSTOM PROMPT ==========
# def get_chain(vectorstore, memory):
#     llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=os.environ["GROQ_API_KEY"])
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": MEDICAL_PROMPT},
#         return_source_documents=False,
#         verbose=False
#     )

# # ========== HANDLE GREETINGS & OUT OF SCOPE (bypass RAG) ==========
# def is_greeting(text):
#     # Only pure greetings without any other words
#     clean = text.lower().strip()
#     greetings = ["hi", "hello", "hey", "salam", "good morning", "good evening", "hola", "hi there"]
#     return clean in greetings or clean.startswith(("hi", "helo")) and len(clean) < 6

# def is_out_of_scope(text):
#     non_medical_keywords = ["graphics", "sports", "weather", "politics", "cooking", "technology", "computer", "python", "code", "programming"]
#     return any(keyword in text.lower() for keyword in non_medical_keywords)

# def get_greeting_response():
#     return "👋 Hello! I'm **MediBot**, your medical AI assistant. How can I help you today? 😊"

# def get_out_of_scope_response():
#     return "🚫 I'm MediBot, a Medical AI Assistant. I can only answer health‑related questions. Please ask me about diseases, symptoms, or treatments! 😊"

# # ========== MAIN APP ==========
# def main():
#     st.set_page_config(page_title="MediBot - Medical AI Assistant", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
#     inject_custom_css()
#     init_session()

#     # Sidebar
#     with st.sidebar:
#         st.markdown("<div style='text-align:center;'><h2>🩺 MediBot</h2><p style='color:#6c757d;'>Your AI Medical Assistant</p></div>", unsafe_allow_html=True)
#         st.markdown("---")
#         st.info("**ℹ️ About**\n- 🏥 Medical info from trusted sources\n- 💬 Symptom checker\n- ⚠️ Emergency detection\n- ✅ Always consult a doctor")
#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.messages = []
#             st.session_state.memory.clear()
#             st.rerun()
#         st.markdown("---")
#         st.caption("Built with Streamlit • Powered by Groq")

#     # Main chat area
#     st.markdown("<h1 style='text-align:center; background: linear-gradient(135deg, #0078FF, #00B4D8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>MediBot</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align:center; color:#6c757d;'>Your trusted medical companion — ask anything about health</p>", unsafe_allow_html=True)

#     # Display chat history
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
#             st.markdown(msg["content"])

#     # Suggestion pills (centered grid)
#     if len(st.session_state.messages) == 0:
#         suggestions = ["🤒 Cold symptoms", "💊 Lower blood pressure", "🫀 Heart attack signs", "🧠 Better sleep", "🌡️ Fever in adults", "✅ Diabetes prevention"]
#         st.markdown('<div class="suggestions-grid">', unsafe_allow_html=True)
#         for s in suggestions:
#             if st.button(s, key=f"sug_{s}", use_container_width=False):
#                 st.session_state.suggestion_clicked = s
#                 st.rerun()
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Process user input
#     prompt = None
#     if 'suggestion_clicked' in st.session_state and st.session_state.suggestion_clicked:
#         prompt = st.session_state.suggestion_clicked
#         st.session_state.suggestion_clicked = None
#     else:
#         prompt = st.chat_input("Ask me about symptoms, diseases, treatments...")

#     if prompt:
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(prompt)

#         # Show typing indicator
#         with st.chat_message("assistant", avatar="🤖"):
#             indicator = st.empty()
#             indicator.markdown('<div class="typing-indicator"><span></span><span></span><span></span><span style="margin-left:8px;">MediBot is thinking...</span></div>', unsafe_allow_html=True)

#         # Route
#         if is_greeting(prompt):
#             answer = get_greeting_response()
#         elif is_out_of_scope(prompt):
#             answer = get_out_of_scope_response()
#         else:
#             try:
#                 vectorstore = get_vectorstore()
#                 chain = get_chain(vectorstore, st.session_state.memory)
#                 # Pass chat history explicitly
#                 chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
#                 response = chain.invoke({"question": prompt, "chat_history": chat_history})
#                 answer = response["answer"]
#             except Exception as e:
#                 answer = f"⚠️ Sorry, I encountered an error: {str(e)}"

#         indicator.empty()
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(answer)
#         st.session_state.messages.append({"role": "assistant", "content": answer})
#         st.rerun()

# if __name__ == "__main__":
#     main()


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

        /* ✅ Suggestions — mobile friendly wrap */
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


# ✅ Fix: chat_history removed from combine_docs prompt — only context + question
MEDICAL_PROMPT = PromptTemplate(
    template="""You are MediBot, a medical AI assistant. Follow these rules strictly:

1. GREETINGS: If user says only "hi", "hello", "hey", "salam", "good morning" → respond warmly, introduce yourself, ask how to help.
2. GRATITUDE & STOP: If user says "thanks", "thank you", "good thanks", "okay fine", "stop", "that's enough" → respond: "You're welcome! 😊 Feel free to return if you have more health questions. Take care!" and do NOT provide medical info.
3. OUT OF SCOPE: If user asks non-medical topics → respond: "🚫 I'm a Medical AI Assistant. I can only answer health-related questions."
4. MEDICAL QUESTIONS: Use the Context to answer. If context lacks info, say "I don't have enough information. Please consult a doctor."
5. For symptoms → list possibilities and add "Please consult a doctor for proper diagnosis."
6. For emergencies (chest pain, breathing trouble, stroke, severe bleeding, seizure) → respond "⚠️ MEDICAL EMERGENCY! Call emergency services immediately."
7. NEVER repeat the same answer twice. If user asks "tell me more", provide additional details.
8. Be empathetic, clear, and use 1-2 emojis per answer.
9. NEVER start a medical answer with "Hi, I'm MediBot".

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
        memory=memory,  # ✅ sirf yahan pass karo — invoke mein nahi
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

    # ✅ Suggestions — mobile friendly (3 cols x 2 rows)
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

        # ✅ Fix: typing indicator + answer INSIDE same chat_message block
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
                    # ✅ Fix: sirf question pass karo — memory chain khud handle karti hai
                    response = chain.invoke({"question": prompt})
                    answer = response["answer"]
                except Exception as e:
                    answer = f"⚠️ Sorry, an error occurred: {str(e)}"

            indicator.empty()
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


if __name__ == "__main__":
    main()