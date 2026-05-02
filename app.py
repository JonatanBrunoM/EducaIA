import streamlit as st
import os
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
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="EducaIA - Resumo Acadêmico", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    # Limpeza básica para evitar erros de caractere no PDF
    texto_limpo = texto.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, txt=texto_limpo)
    return pdf.output(dest='S')

# 2. CSS
st.markdown(f"""
    <style>
    .sidebar-top-button {{
        padding-top: 10px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
    }}
    [data-testid="stSidebarCollapseByArrow"] svg {{ display: none; }}
    [data-testid="stSidebarCollapseByArrow"]::after {{
        content: "☰"; font-size: 24px; font-weight: bold; display: flex; justify-content: center; align-items: center;
    }}
    .faculdade-logo {{
        position: absolute; top: -55px; left: 50px; width: 150px; z-index: 99;
    }}
    .sidebar-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .sidebar-logo {{ width: 35px; height: auto; }}
    .stButton > button {{
        border-radius: 20px; border: 1px solid #444746; width: 100%; text-align: left; padding: 10px 20px;
    }}
    .stButton > button:hover {{ border-color: #1e86c8; color: #1e86c8; }}
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sugestao_clicada" not in st.session_state:
    st.session_state.sugestao_clicada = None
if "ultimo_resumo" not in st.session_state:
    st.session_state.ultimo_resumo = None

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown(f'<div class="sidebar-header"><img src="data:image/png;base64,{bin_str_mini}" class="sidebar-logo"><h1 style="font-size: 22px; margin: 0;">EducaIA</h1></div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; opacity: 0.7; margin-bottom: 0;'>Assistente Acadêmico Digital</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-top-button">', unsafe_allow_html=True)
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.session_state.ultimo_resumo = None
        st.rerun()
    
    # FUNCIONALIDADE 2: BOTÃO DE RESUMO
    if st.button("📄 Gerar Resumo para Download"):
        st.session_state.sugestao_clicada = "Gere um resumo estruturado e detalhado dos pontos principais dos documentos para exportação em PDF."
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Sugestões")
    sugestoes = {
        "📑 Evolução das Tecnologias": "Fale sobre a evolução das tecnologias digitais na gestão em saúde.",
        "📑 Cibercultura e suas relações": "Fale sobre a discussão sobre a cibercultura e suas relações com a educação e a saúde.",
        "📑 Aplicativos na Saúde": "Fale sobre os aplicativos utilizados na área da saúde com exemplos e benefícios.",
        "📑 Tecnologias emergentes": "Fale sobre a inteligência artificial, robótica e IoT na saúde."
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

if not st.session_state.messages:
    st.markdown(f'<div class="welcome-text"><h1 class="welcome-title">Olá! Eu sou o EducaIA</h1><p style="font-size: 20px; opacity: 0.8;">Como posso ajudar nos teus estudos hoje?</p></div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = AVATAR_AI if message["role"] == "assistant" else AVATAR_USER
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "image_url" in message:
            if isinstance(message["image_url"], list):
                cols = st.columns(len(message["image_url"]))
                for idx, url in enumerate(message["image_url"]): cols[idx].image(url)
            else:
                st.image(message["image_url"])

input_usuario = st.chat_input("Pergunte algo ou peça um resumo...")
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
                
                img_urls_list = []
                
                # 1. BUSCA DE IMAGEM
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
                            st.markdown(f"Encontrei estas imagens sobre '{prompt_final}':")
                            cols = st.columns(len(img_urls_list))
                            for idx, url in enumerate(img_urls_list): cols[idx].image(url, use_column_width=True)
                            st.session_state.messages.append({"role": "assistant", "content": f"Imagens sobre {prompt_final}", "image_url": img_urls_list})
                    except: pass

                # 2. RESPOSTA TEXTUAL + GLOSSÁRIO
                if not img_urls_list:
                    # Prompt que solicita o GLOSSÁRIO ao final
                    prompt_template = ChatPromptTemplate.from_template(
                        "Você é um tutor amigável. Responda em PT-BR usando o contexto: {context}\n"
                        "Ao final da sua explicação, se houver termos técnicos complicados, "
                        "adicione uma seção chamada '📚 Glossário de Termos' explicando-os brevemente.\n"
                        "Pergunta: {input}"
                    )
                    chain = create_retrieval_chain(base.as_retriever(), create_stuff_documents_chain(llm, prompt_template))
                    response = chain.invoke({"input": prompt_final})
                    full_text = response["answer"]
                    
                    st.markdown(full_text)
                    st.session_state.messages.append({"role": "assistant", "content": full_text})
                    
                    # Se foi pedido resumo, guarda para o botão de PDF
                    if "resumo" in prompt_final.lower():
                        st.session_state.ultimo_resumo = full_text

                # Exibe botão de download se houver resumo
                if st.session_state.ultimo_resumo:
                    pdf_data = gerar_pdf_resumo(st.session_state.ultimo_resumo)
                    st.download_button(
                        label="📥 Baixar Resumo em PDF",
                        data=pdf_data,
                        file_name="resumo_educaia.pdf",
                        mime="application/pdf"
                    )

                if len(st.session_state.messages) <= 2: st.rerun()
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
