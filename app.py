import streamlit as st
import os
import base64
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuração da Página
st.set_page_config(
    page_title="EducaIA", 
    page_icon="logomini.png", 
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

# 2. CSS Customizado
st.markdown(f"""
    <style>
    .sidebar-top-button {{
        padding-top: 10px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
    }}
    .faculdade-logo {{
        position: absolute; top: -55px; left: 50px; width: 150px; z-index: 99;
    }}
    .sidebar-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .sidebar-logo {{ width: 35px; height: auto; }}
    .stButton > button {{
        border-radius: 20px; border: 1px solid #444746; width: 100%; text-align: left; padding: 10px 20px;
    }}
    .stDeployButton {{display:none;}}
    footer {{visibility: hidden;}}
    .welcome-text {{ text-align: center; margin-top: 15vh; }}
    .welcome-title {{
        font-size: 50px; font-weight: 600; background: linear-gradient(90deg, #1e86c8, #8ac5e2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE IA ---
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

# Inicialização de estados
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sugestao_clicada" not in st.session_state:
    st.session_state.sugestao_clicada = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown(f'<div class="sidebar-header"><img src="data:image/png;base64,{bin_str_mini}" class="sidebar-logo"><h1 style="font-size: 22px; margin: 0;">EducaIA</h1></div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; opacity: 0.7; margin-bottom: 0;'>Assistente Acadêmico Digital</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-top-button">', unsafe_allow_html=True)
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.session_state.quiz_data = None
        st.rerun()
    
    # FUNCIONALIDADE 1: GERAR QUIZ
    if st.button("🧠 Gerar Quiz Interativo"):
        st.session_state.quiz_data = None # Reseta quiz anterior
        st.session_state.sugestao_clicada = "Gere um quiz de múltipla escolha. Formato: PERGUNTA: [texto] | A) [opção] | B) [opção] | C) [opção] | CORRETA: [letra]"
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Sugestões")
    sugestoes = {
        "📑 Evolução das Tecnologias": "Fale sobre a evolução das tecnologias digitais na gestão em saúde.",
        "📑 Aplicativos na Saúde": "Fale sobre os aplicativos utilizados na área da saúde.",
        "📑 Tecnologias Emergentes": "Fale sobre a introdução às tecnologias emergentes na saúde."
    }
    for label, prompt in sugestoes.items():
        if st.button(label): st.session_state.sugestao_clicada = prompt

# --- ÁREA PRINCIPAL ---
st.markdown(f'<img src="data:image/png;base64,{bin_str_faculdade}" class="faculdade-logo">', unsafe_allow_html=True)

base = processar_base()
if not base:
    st.error("Banco de dados não localizado.")
    st.stop()

AVATAR_USER = "👤"
AVATAR_AI = f"data:image/png;base64,{bin_str_mini}"

if not st.session_state.messages and not st.session_state.quiz_data:
    st.markdown(f'<div class="welcome-text"><h1 class="welcome-title">Olá! Eu sou o EducaIA</h1><p style="font-size: 20px; opacity: 0.8;">Vamos transformar seus PDFs em conhecimento?</p></div>', unsafe_allow_html=True)

# Exibição do Histórico (sem repetir o quiz ativo no histórico para não bugar)
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATAR_AI if message["role"] == "assistant" else AVATAR_USER):
        st.markdown(message["content"])

input_usuario = st.chat_input("Pergunte algo...")
prompt_final = input_usuario if input_usuario else st.session_state.sugestao_clicada
if st.session_state.sugestao_clicada: st.session_state.sugestao_clicada = None

if prompt_final:
    st.session_state.messages.append({"role": "user", "content": prompt_final})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt_final)

    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Buscando informações..."):
            try:
                chave_groq = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(groq_api_key=chave_groq, model_name="llama-3.1-8b-instant", temperature=0.4)
                
                # RAG
                prompt_template = ChatPromptTemplate.from_template("Use o contexto: {context}\nPergunta: {input}")
                chain = create_retrieval_chain(base.as_retriever(), create_stuff_documents_chain(llm, prompt_template))
                response = chain.invoke({"input": prompt_final})
                resposta_texto = response["answer"]

                # Lógica de Captura do Quiz
                if "|" in resposta_texto and "PERGUNTA:" in resposta_texto.upper():
                    partes = resposta_texto.split("|")
                    if len(partes) >= 5:
                        pergunta = partes[0].replace("PERGUNTA:", "").strip()
                        opcoes = [partes[1].strip(), partes[2].strip(), partes[3].strip()]
                        correta_texto = partes[4].upper()
                        # Extrai apenas a letra A, B ou C
                        correta = "A" if "A" in correta_texto else "B" if "B" in correta_texto else "C"
                        
                        st.session_state.quiz_data = {"p": pergunta, "o": opcoes, "c": correta}
                        st.markdown("Preparei um quiz para você abaixo:")
                    else:
                        st.markdown(resposta_texto)
                else:
                    st.markdown(resposta_texto)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
                
                st.rerun() # Força o Streamlit a renderizar o rádio do quiz
            except Exception as e:
                st.error(f"Erro: {e}")

# Renderização do Quiz Ativo (fora do loop de mensagens para ser interativo)
if st.session_state.quiz_data:
    with st.chat_message("assistant", avatar=AVATAR_AI):
        q = st.session_state.quiz_data
        st.markdown(f"### 📝 DESAFIO\n**{q['p']}**")
        
        escolha = st.radio("Selecione a opção correta:", q['o'], index=None, key="radio_quiz")
        
        if escolha:
            letra_escolhida = escolha[0].upper()
            if letra_escolhida == q['c']:
                st.success(f"✅ Correto! A resposta certa é a {q['c']}.")
            else:
                st.error(f"❌ Incorreto. Você marcou {letra_escolhida}, mas a resposta certa é a {q['c']}.")
            
            if st.button("Finalizar e Salvar"):
                resultado = f"Quiz Respondido: {q['p']} (Correta: {q['c']})"
                st.session_state.messages.append({"role": "assistant", "content": resultado})
                st.session_state.quiz_data = None
                st.rerun()
