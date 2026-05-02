import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuração da Página
st.set_page_config(page_title="EducaIA", page_icon="✨", layout="wide")

# 2. CSS Estilo "Gemini" com ajuste de cor no Título
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1e1f20;
        border-right: 1px solid #333;
    }
    
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1 {
        color: #e3e3e3 !important;
    }

    .stButton > button {
        border-radius: 20px;
        background-color: #2b2c2e;
        color: #c4c7c5;
        border: 1px solid #444746;
        width: 100%;
        text-align: left;
        padding: 10px 20px;
        margin-bottom: 5px;
    }
    
    .stButton > button:hover {
        background-color: #333537;
        border-color: #a8c7fa;
        color: #a8c7fa;
    }

    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    
    /* --- ALTERE AQUI: Estilização do Título de Boas-vindas --- */
    .welcome-text {
        text-align: center;
        margin-top: 15vh;
    }
    
    /* Aqui definimos um gradiente para o título não sumir no fundo escuro */
    .welcome-title {
        font-size: 50px;
        font-weight: 600;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .welcome-subtitle {
        font-size: 22px;
        color: #888; /* Cor cinza suave que funciona em ambos os temas */
    }
    /* -------------------------------------------------------- */
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA BIBLIOTECA ---
# ALTERE AQUI: Adicione novos nomes de arquivos nesta lista
LIVROS = ["ebook1.pdf", "ebook2.pdf", "ebook3.pdf", "ebook4.pdf"]

@st.cache_resource
def processar_base():
    documentos = []
    for arquivo in LIVROS:
        if os.path.exists(arquivo):
            loader = PyPDFLoader(arquivo)
            documentos.extend(loader.load())
    if not documentos: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    textos = text_splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(textos, embeddings)

# --- INICIALIZAÇÃO DO ESTADO ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sugestao_clicada" not in st.session_state:
    st.session_state.sugestao_clicada = None

# --- BARRA LATERAL ---
with st.sidebar:
    # ALTERE AQUI: Título da Sidebar
    st.markdown("<h1 style='font-size: 25px;'>✨ EducaIA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px;'>Seu assistente acadêmico</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Sugestões")
    
    # ALTERE AQUI: Texto dos botões e a pergunta que eles disparam
    if st.button("🚀 Evolução das Tecnologias"):
        st.session_state.sugestao_clicada = "Fale sobre a evolução das tecnologias digitais na gestão em saúde."

    if st.button("📑 Conceitos Chave"):
        st.session_state.sugestao_clicada = "Quais os conceitos fundamentais do material?"

    if st.button("📝 Simulado"):
        st.session_state.sugestao_clicada = "Crie 3 questões de múltipla escolha para eu treinar."

    # Botão de limpar no rodapé da sidebar
    st.markdown("<div style='position: fixed; bottom: 20px; width: 260px;'>", unsafe_allow_html=True)
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- MOTOR DE IA ---
if any(os.path.exists(f) for f in LIVROS):
    base = processar_base()
else:
    st.error("Banco de dados não localizado.")
    st.stop()

# --- ÁREA DE CHAT ---

# ALTERE AQUI: Conteúdo da tela de boas-vindas
if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-text">
            <h1 class="welcome-title">Olá! Eu sou o EducaIA</h1>
            <p class="welcome-subtitle">Como posso ajudar nos seus estudos hoje?</p>
        </div>
        """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input_usuario = st.chat_input("Digite uma pergunta...")

# Lógica de resposta
prompt_final = None
if st.session_state.sugestao_clicada:
    prompt_final = st.session_state.sugestao_clicada
    st.session_state.sugestao_clicada = None
elif input_usuario:
    prompt_final = input_usuario

if prompt_final:
    st.session_state.messages.append({"role": "user", "content": prompt_final})
    with st.chat_message("user"):
        st.markdown(prompt_final)

    try:
        chave = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(groq_api_key=chave, model_name="llama-3.1-8b-instant", temperature=0.3)
        
        # ALTERE AQUI: Instruções de personalidade da IA
        prompt_template = ChatPromptTemplate.from_template("""
        Você é o EducaIA, um tutor especializado. Use o contexto para responder de forma clara.
        Contexto: {context}
        Pergunta: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(base.as_retriever(), document_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = retrieval_chain.invoke({"input": prompt_final})
                texto_resposta = response["answer"]
                st.markdown(texto_resposta)
                st.session_state.messages.append({"role": "assistant", "content": texto_resposta})
                
        if len(st.session_state.messages) <= 2:
            st.rerun()

    except Exception as e:
        st.error(f"Erro na conexão: {e}")
