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


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print("Carregando documentos da pasta /pdfs...")
loader = DirectoryLoader("pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
pdfs = loader.load()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
spliter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    chunk_size=1200,
    tokenizer=tokenizer,
    chunk_overlap=120
)

chunks = spliter.split_documents(pdfs)
print(f"Documentos divididos em {len(chunks)} partes.")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=api_key
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Criando banco vetorial...")
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Engenheiro de Software especializado em arquitetura de microsserviços. "
     "Use o contexto abaixo para responder. "
     "Se não souber, diga que não sabe.\n\n{context}"),
    ("human", "{input}"),
])

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def perguntar(pergunta: str):
    print(f"\n--- Pergunta: {pergunta} ---")
    return rag_chain.invoke(pergunta)

if __name__ == "__main__":
    if chunks:
        print("\nChatbot pronto! Digite 'sair' para encerrar.\n")
        while True:
            pergunta = input("Você: ")
            if pergunta.lower() == "sair":
                break
            resposta = perguntar(pergunta)
            print("\nResposta da IA:", resposta)
    else:
        print("Aviso: Nenhum PDF encontrado na pasta 'pdfs'.")