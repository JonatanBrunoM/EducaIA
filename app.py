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
st.set_page_config(page_title="EducaIA | Tutor Inteligente", page_icon="🤖", layout="wide")

# 2. CSS Personalizado para Cor da Sidebar e Efeito Centralizado
st.markdown("""
    <style>
    /* Cor de fundo da Sidebar (Azul Escuro Profissional) */
    [data-testid="stSidebar"] {
        background-color: #1E1E2F;
        color: white;
    }
    
    /* Ajuste da cor dos textos na Sidebar */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* Estilização dos botões da Sidebar */
    div.stButton > button {
        background-color: #3D3D5C;
        color: white;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #57578A;
        color: #00FFCC;
    }

    /* Centralização do Título quando não há chat */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
    }

    /* Esconder o botão de deploy padrão */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
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

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("## 🤖 EducaIA")
    st.caption("Versão 2.0 - Banco de Dados Acadêmico")
    st.markdown("---")
    
    st.subheader("💡 Sugestões")
    
    if st.button("🚀 Evolução das Tecnologias"):
        st.session_state.sugestao_clicada = "Fale sobre a evolução das tecnologias digitais na gestão em saúde."

    if st.button("📑 Conceitos Chave"):
        st.session_state.sugestao_clicada = "Quais os conceitos fundamentais do material?"

    if st.button("📝 Simulado"):
        st.session_state.sugestao_clicada = "Crie 3 questões de múltipla escolha para eu treinar."

    st.markdown("---")
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MOTOR DE INTELIGÊNCIA ---
if any(os.path.exists(f) for f in LIVROS):
    base = processar_base()
else:
    st.error("Banco de dados não localizado.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Captura clique de sugestão
prompt_final = None
if "sugestao_clicada" in st.session_state and st.session_state.sugestao_clicada:
    prompt_final = st.session_state.sugestao_clicada
    st.session_state.sugestao_clicada = None 

# --- ÁREA CENTRAL / CHAT ---

# Se não houver mensagens, mostra o título centralizado
if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-container">
            <h1 style='font-size: 3rem;'>📚 Como posso ajudar?</h1>
            <p style='font-size: 1.2rem; color: #666;'>Consulte o material da disciplina agora mesmo.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.title("📚 Central de Conhecimento")

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input de chat (sempre no rodapé por padrão do Streamlit)
input_usuario = st.chat_input("Digite sua pergunta...")

if input_usuario or prompt_final:
    texto_da_pergunta = input_usuario if input_usuario else prompt_final
    st.session_state.messages.append({"role": "user", "content": texto_da_pergunta})
    
    # Recarrega para aplicar a mudança de layout (sair do centro)
    if len(st.session_state.messages) == 1:
        st.rerun()

    with st.chat_message("user"):
        st.markdown(texto_da_pergunta)

    try:
        chave = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(groq_api_key=chave, model_name="llama-3.1-8b-instant", temperature=0.3)
        
        prompt_template = ChatPromptTemplate.from_template("""
        Você é o EducaIA, um tutor acadêmico especializado.
        Responda ESTRITAMENTE com base no contexto:
        {context}
        Pergunta: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(base.as_retriever(), document_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Analisando materiais..."):
                response = retrieval_chain.invoke({"input": texto_da_pergunta})
                texto_resposta = response["answer"]
                st.markdown(texto_resposta)
        st.session_state.messages.append({"role": "assistant", "content": texto_resposta})
    except Exception as e:
        st.error(f"Erro: {e}")
