import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NOVA FORMA DE IMPORTAR A CHAIN (Evita o erro ModuleNotFoundError)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="EducaIA - Assistente", layout="centered")

# Certifique-se de que esses nomes são IGUAIS aos arquivos no GitHub
LIVROS = ["ebook1.pdf", "ebook2.pdf", "ebook3.pdf", "ebook4.pdf"]

@st.cache_resource
def processar_base():
    documentos = []
    for arquivo in LIVROS:
        if os.path.exists(arquivo):
            loader = PyPDFLoader(arquivo)
            documentos.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(textos, embeddings)

st.title("📚 EducaIA")

if all(os.path.exists(f) for f in LIVROS):
    with st.status("Lendo ebooks...", expanded=False) as status:
        base = processar_base()
        status.update(label="Base pronta!", state="complete")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Dúvida sobre os PDFs?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            chave = st.secrets["GROQ_API_KEY"]
            llm = ChatGroq(groq_api_key=chave, model_name="llama3-8b-8192")
            
            # Definindo como a IA deve se comportar
            prompt_template = ChatPromptTemplate.from_template("""
            Responda à pergunta com base apenas no contexto fornecido (os ebooks):
            <context>
            {context}
            </context>
            Pergunta: {input}""")

            # Criando a corrente de documentos e a de recuperação
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(base.as_retriever(), document_chain)
            
            with st.chat_message("assistant"):
                with st.spinner("Analisando livros..."):
                    # Executando a busca e resposta
                    response = retrieval_chain.invoke({"input": prompt})
                    texto_resposta = response["answer"]
                    st.markdown(texto_resposta)
            
            st.session_state.messages.append({"role": "assistant", "content": texto_resposta})
        except Exception as e:
            st.error(f"Erro na consulta: {e}")
else:
    st.error("ERRO: Os PDFs não foram encontrados. Verifique se os nomes no GitHub estão como livro1.pdf, etc.")
