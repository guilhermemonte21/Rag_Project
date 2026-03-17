import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot RAG - Engenheiro de Software")

# Inicialização do RAG com cache para não carregar os PDFs toda hora
@st.cache_resource(show_spinner="Carregando documentos e inicializando a IA...")
def init_rag():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Por favor, defina a GOOGLE_API_KEY no arquivo .env")
        st.stop()
        
    loader = DirectoryLoader("pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
    pdfs = loader.load()
    
    if not pdfs:
        st.warning("Aviso: Nenhum PDF encontrado na pasta 'pdfs'. Adicione PDFs para testar.")
        return None
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    spliter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        chunk_size=1200,
        tokenizer=tokenizer,
        chunk_overlap=120
    )
    
    chunks = spliter.split_documents(pdfs)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um Engenheiro de Software especializado em arquitetura de microsserviços. "
         "Use o contexto abaixo para responder. "
         "Se não souber, diga que não sabe.\n\n{context}"),
        MessagesPlaceholder("history"), 
        ("human", "{input}"),
    ])
    
    def get_context(input_dict):
        pergunta = input_dict["input"] if isinstance(input_dict, dict) else input_dict
        docs = retriever.invoke(pergunta)
        return format_docs(docs)
        
    rag_chain = (
        RunnablePassthrough.assign(context=get_context)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Inicialização do histórico na sessão do Streamlit
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

# Carrega o modelo
rag_chain = init_rag()

if rag_chain is None:
    st.stop()

# Monta a chain com histórico
rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Exibe as mensagens já enviadas na interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Captura de input do usuário
if prompt_msg := st.chat_input("Faça sua pergunta:"):
    # Adiciona a pergunta do usuário no histórico da UI
    st.session_state.messages.append({"role": "user", "content": prompt_msg})
    with st.chat_message("user"):
        st.markdown(prompt_msg)
        
    # Gera e exibe a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resposta = rag_chain_with_history.invoke(
                {"input": prompt_msg},
                config={"configurable": {"session_id": "chat1"}}
            )
            st.markdown(resposta)
            
    # Adiciona a resposta da IA no histórico da UI
    st.session_state.messages.append({"role": "assistant", "content": resposta})
