import streamlit as st
import os
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- ALTERE AQUI: Configuração do Favicon ---
# Usamos o arquivo local 'logomini.png' como o ícone da aba
st.set_page_config(
    page_title="EducaIA", 
    page_icon="logomini.png", # Aqui definimos o favicon
    layout="wide"
)

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

bin_str_mini = get_base64_of_bin_file('logomini.png')
bin_str_faculdade = get_base64_of_bin_file('logofaculdade.png')

# 2. CSS Estilo "Gemini" + Ajuste do ícone da Sidebar
st.set_page_config(
    page_title="EducaIA", 
    page_icon="logomini.png", 
    layout="wide"
)

# --- NO CSS (Início do código) ---
st.markdown(f"""
    <style>
    /* FORÇANDO O ÍCONE DE HAMBÚRGUER */
    /* Este seletor tenta esconder a seta e colocar o ícone de menu do Material Design */
    [data-testid="stSidebarCollapseByArrow"] svg {{
        display: none;
    }}
    [data-testid="stSidebarCollapseByArrow"]::after {{
        content: "☰"; /* Símbolo de hambúrguer */
        color: white;
        font-size: 24px;
        font-weight: bold;
        display: flex;
        justify-content: center;
        align-items: center;
        padding-left: 10px;
    }}

    /* Ajuste de margem para a logo da faculdade não cobrir o menu */
    .faculdade-logo {{
        position: absolute;
        top: -60px;
        left: 50px; /* Aumentei um pouco para dar espaço ao botão de menu */
        width: 150px;
        z-index: 99;
    }}

    .sidebar-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }}
    .sidebar-logo {{
        width: 35px;
        height: auto;
    }}

    .stButton > button {{
        border-radius: 20px;
        background-color: #2b2c2e;
        color: #c4c7c5;
        border: 1px solid #444746;
        width: 100%;
        text-align: left;
        padding: 10px 20px;
    }}
    
    .stButton > button:hover {{
        background-color: #333537;
        border-color: #a8c7fa;
        color: #a8c7fa;
    }}

    .stDeployButton {{display:none;}}
    footer {{visibility: hidden;}}
    
    .welcome-text {{
        text-align: center;
        margin-top: 15vh;
    }}
    
    .welcome-title {{
        font-size: 50px;
        font-weight: 600;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA BIBLIOTECA ---
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sugestao_clicada" not in st.session_state:
    st.session_state.sugestao_clicada = None

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-header">
            <img src="data:image/png;base64,{bin_str_mini}" class="sidebar-logo">
            <h1 style='font-size: 22px; margin: 0; color: white;'>EducaIA</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 14px; color: #aaa;'>Assistente Acadêmico Digital</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Sugestões")
    if st.button("🚀 Evolução das Tecnologias"):
        st.session_state.sugestao_clicada = "Fale sobre a evolução das tecnologias digitais na gestão em saúde."
    if st.button("📑 Conceitos Chave"):
        st.session_state.sugestao_clicada = "Quais os conceitos fundamentais do material?"
    if st.button("📝 Simulado"):
        st.session_state.sugestao_clicada = "Crie 3 questões de múltipla escolha para eu treinar."

    st.markdown("<div style='position: fixed; bottom: 20px; width: 260px;'>", unsafe_allow_html=True)
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- ÁREA PRINCIPAL ---
st.markdown(f'<img src="data:image/png;base64,{bin_str_faculdade}" class="faculdade-logo">', unsafe_allow_html=True)

if any(os.path.exists(f) for f in LIVROS):
    base = processar_base()
else:
    st.error("Banco de dados não localizado.")
    st.stop()

if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-text">
            <h1 class="welcome-title">Olá! Eu sou o EducaIA</h1>
            <p style="font-size: 20px; color: #888;">Como posso ajudar nos seus estudos hoje?</p>
        </div>
        """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input_usuario = st.chat_input("Pergunta ao EducaIA...")

# Lógica de processamento
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
        prompt_template = ChatPromptTemplate.from_template("Responda em PT-BR usando o contexto: {context}\nPergunta: {input}")
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(base.as_retriever(), document_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Pesquisando nos ebooks..."):
                response = retrieval_chain.invoke({"input": prompt_final})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        
        if len(st.session_state.messages) <= 2: st.rerun()
    except Exception as e:
        st.error(f"Erro: {e}")
