import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Configuração da página para ficar bonita
st.set_page_config(page_title="EducaIA - Assistente de Estudos", layout="centered")

# LISTA DOS LIVROS (Verifique se os nomes batem com os que você subiu)
LIVROS = ["ebook1.pdf", "ebook2.pdf", "ebook3.pdf", "ebook4.pdf"]

@st.cache_resource
def processar_base():
    documentos = []
    for arquivo in LIVROS:
        if os.path.exists(arquivo):
            loader = PyPDFLoader(arquivo)
            documentos.extend(loader.load())
    
    # Divide o texto em blocos menores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos)
    
    # Cria os Embeddings (transforma texto em números)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Cria o banco de dados vetorial
    return FAISS.from_documents(textos, embeddings)

# --- INTERFACE ---
st.title("📚 EducaIA")
st.subheader("Sua inteligência artificial acadêmica")

# Verifica se os arquivos estão lá
if all(os.path.exists(f) for f in LIVROS):
    with st.status("Preparando base de conhecimento...", expanded=False) as status:
        base = processar_base()
        status.update(label="Base de dados pronta!", state="complete", expanded=False)

    # Inicializa o histórico do chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Campo de pergunta
    if prompt := st.chat_input("Pergunte algo sobre a matéria:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # CHAVE DA GROQ (Recomendo colocar diretamente aqui para o trabalho)
        # Substitua SUA_CHAVE pela chave que você gerou
      minha_chave = st.secrets["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"]
        llm = ChatGroq(groq_api_key=minha_chave, model_name="llama3-8b-8192")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=base.as_retriever()
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Consultando livros..."):
                resposta = qa_chain.run(prompt)
                st.markdown(resposta)
        
        st.session_state.messages.append({"role": "assistant", "content": resposta})
else:
    st.error("Erro: Não encontrei os PDFs no repositório. Verifique os nomes dos arquivos.")
