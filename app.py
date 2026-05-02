import streamlit as st
import os
import io
import base64
import requests
from fpdf import FPDF  # Necessário: pip install fpdf2
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

# --- FUNÇÃO GERADORA DE PDF ---
def gerar_pdf_resumo(texto):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, txt="EducaIA - Resumo da Conversa", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    texto_limpo = texto.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, txt=texto_limpo)
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes) if isinstance(pdf_bytes, bytearray) else pdf_bytes

# 2. CSS
st.markdown(f"""
    <style>
    .sidebar-top-button {{ padding-top: 10px; padding-bottom: 20px; border-bottom: 1px solid rgba(128, 128, 128, 0.2); margin-bottom: 20px; }}
    .faculdade-logo {{ position: absolute; top: -55px; left: 50px; width: 150px; z-index: 99; }}
    .stButton > button {{ border-radius: 20px; border: 1px solid #444746; width: 100%; text-align: left; padding: 10px 20px; }}
    .welcome-text {{ text-align: center; margin-top: 15vh; }}
    .welcome-title {{ font-size: 50px; font-weight: 600; background: linear-gradient(90deg, #1e86c8, #8ac5e2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sugestao_clicada" not in st.session_state:
    st.session_state.sugestao_clicada = None
if "ultimo_resumo" not in st.session_state:
    st.session_state.ultimo_resumo = None

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown(f'# EducaIA')
    st.markdown('<div class="sidebar-top-button">', unsafe_allow_html=True)
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.session_state.ultimo_resumo = None
        st.rerun()
    
    if st.button("📄 Baixar conversa"):
        if len(st.session_state.messages) > 0:
            conteudo_chat = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages if "image_url" not in m])
            st.session_state.sugestao_clicada = f"Com base exclusivamente na nossa conversa abaixo, crie um resumo estruturado para meus estudos:\n\n{conteudo_chat}"
        else:
            st.warning("Inicie uma conversa para poder resumir!")
    st.markdown('</div>', unsafe_allow_html=True)

# --- ÁREA PRINCIPAL ---
st.markdown(f'<img src="data:image/png;base64,{bin_str_faculdade}" class="faculdade-logo">', unsafe_allow_html=True)
base = processar_base()

AVATAR_USER = "👤"
AVATAR_AI = f"data:image/png;base64,{bin_str_mini}"

if not st.session_state.messages:
    st.markdown(f'<div class="welcome-text"><h1 class="welcome-title">Olá! Eu sou o EducaIA</h1><p style="font-size: 20px; opacity: 0.8;">O que vamos aprender hoje?</p></div>', unsafe_allow_html=True)

# EXIBIÇÃO DO HISTÓRICO (COM SUPORTE A IMAGENS)
for message in st.session_state.messages:
    avatar = AVATAR_AI if message["role"] == "assistant" else AVATAR_USER
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "image_urls" in message:
            cols = st.columns(len(message["image_urls"]))
            for idx, url in enumerate(message["image_urls"]):
                cols[idx].image(url, use_container_width=True)

input_usuario = st.chat_input("Pergunte algo...")
prompt_final = input_usuario if input_usuario else st.session_state.sugestao_clicada
if st.session_state.sugestao_clicada: st.session_state.sugestao_clicada = None

if prompt_final:
    st.session_state.messages.append({"role": "user", "content": prompt_final})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt_final)

    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Processando..."):
            try:
                chave_groq = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(groq_api_key=chave_groq, model_name="llama-3.1-8b-instant", temperature=0.3)
                
                # --- 1. LÓGICA DE BUSCA DE IMAGEM ---
                img_urls_list = []
                if any(x in prompt_final.lower() for x in ["imagem", "foto", "mostre", "veja", "figura"]):
                    try:
                        serper_key = st.secrets["SERPER_API_KEY"]
                        url_serper = "https://google.serper.dev/images"
                        payload = {"q": prompt_final, "num": 3}
                        headers = {'X-API-KEY': serper_key, 'Content-Type': 'application/json'}
                        response_serper = requests.post(url_serper, headers=headers, json=payload)
                        search_results = response_serper.json()
                        if search_results.get('images'):
                            img_urls_list = [img['imageUrl'] for img in search_results['images']]
                    except: pass

                # --- 2. LÓGICA DE TEXTO ---
                if "nossa conversa abaixo" in prompt_final:
                    resposta_texto = llm.invoke(prompt_final).content
                    st.session_state.ultimo_resumo = resposta_texto
                else:
                    prompt_template = ChatPromptTemplate.from_template(
                        "Você é um tutor amigável. Responda em PT-BR usando o contexto: {context}\n"
                        "Ao final, se houver termos técnicos, adicione um '📚 Glossário'.\n"
                        "Pergunta: {input}"
                    )
                    chain = create_retrieval_chain(base.as_retriever(), create_stuff_documents_chain(llm, prompt_template))
                    response = chain.invoke({"input": prompt_final})
                    resposta_texto = response["answer"]

                # Exibe as imagens se houver
                if img_urls_list:
                    st.markdown(f"Encontrei estas imagens sobre '{prompt_final}':")
                    cols = st.columns(len(img_urls_list))
                    for idx, url in enumerate(img_urls_list):
                        cols[idx].image(url, use_container_width=True)
                
                # Exibe o texto
                st.markdown(resposta_texto)
                
                # Salva no histórico (Texto + Imagens)
                history_entry = {"role": "assistant", "content": resposta_texto}
                if img_urls_list:
                    history_entry["image_urls"] = img_urls_list
                st.session_state.messages.append(history_entry)

                st.rerun()
            except Exception as e:
                st.error(f"Erro ao processar: {e}")

if st.session_state.ultimo_resumo:
    st.divider()
    pdf_data = gerar_pdf_resumo(st.session_state.ultimo_resumo)
    st.download_button(
        label="📥 Baixar Resumo da Conversa em PDF",
        data=pdf_data,
        file_name="meu_estudo_educaia.pdf",
        mime="application/pdf",
        use_container_width=True
    )
