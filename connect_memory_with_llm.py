import os
from dotenv import load_dotenv, find_dotenv   # type: ignore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain.chains import RetrievalQA # type: ignore

load_dotenv(find_dotenv())

# Phase 2–Connect Memory with LLM
# ● Step 1: Setup LLM (Mistral with HuggingFace)


HF_TOKEN= os.getenv("HF_TOKEN")
HUGGINFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def get_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512
    )
    return llm



# ● Step 2: Connect LLM with FAISS


CUSTOM_PROMPT_TEMPLATE = """
You are a knowledgeable and careful medical AI assistant.

Your task is to answer the user's question strictly based on the provided context and previous conversation history.

Instructions:
1. Use ONLY the information given in the context and chat history.
2. Do NOT use outside knowledge or make assumptions.
3. If the answer is not available, respond with: "I don't know."

Understanding User Intent:
- If the user asks about a disease → explain it clearly (definition, causes, symptoms).
- If the user wants more details → provide a deeper and easy-to-understand explanation.
- If the user asks for treatment → explain step-by-step treatment options based only on the context.
- If the user provides symptoms → relate them to possible conditions ONLY from the context.

Response Rules:
- Keep the answer clear, structured, and easy to understand.
- Do not give too little or too much information.
- Ensure the user's question is fully answered.
- Do NOT include small talk or unnecessary explanations.
- Follow the exact instruction given by the user (format/style if specified).

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(custom_prompt_templete):
    prompt = PromptTemplate(template=custom_prompt_templete, input_variables=["context", "question"])
    return prompt

# Load dataset from FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model , allow_dangerous_deserialization=True)


# ● Step 3: Create chain

qa_chain = RetrievalQA.from_chain_type(
    llm = get_llm(huggingface_repo_id=HUGGINFACE_REPO_ID, hf_token=HF_TOKEN),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}), # Adjust 'k' to control the number of retrieved documents
    return_source_documents = True,
    chain_type_kwargs = {"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke the chain with a question to test the connection between LLM and FAISS

prompt = input("Ask a medical question: ")
response = qa_chain.invoke({"query": prompt}) 
print("Answer: ", response["result"])
print("Source Documents: ", response["source_documents"])