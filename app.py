import streamlit as st
import os
import base64
import requests  # Nova biblioteca para chamar a API do Google
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

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown(f'<div class="sidebar-header"><img src="data:image/png;base64,{bin_str_mini}" class="sidebar-logo"><h1 style="font-size: 22px; margin: 0;">EducaIA</h1></div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; opacity: 0.7; margin-bottom: 0;'>Assistente Acadêmico Digital</p>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-top-button">', unsafe_allow_html=True)
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Sugestões")

    sugestoes = {

        "📑 Evolução das Tecnologias": "Fale sobre a evolução das tecnologias digitais na gestão em saúde.",

        "📑 Incorporação de tecnologias": "Fale sobre a exploração da evolução histórica da incorporação de tecnologias da informação na saúde.",

        "📑 Destaque dos principais marcos": "Fale sobre os os principais marcos e avanços da evolução histórica das tecnologias da informação na saúde.",

        "📑 Cibercultura e suas relações": "Fale sobre a discussão sobre a cibercultura e suas relações com a educação e a saúde.",

        "📑 Princípios básicos da cibercultura": "Aborde os princípios básicos da cibercultura.",

        "📑 Características e fluxos de comunicação": "Fale sobre características e fluxos de comunicação.",

        "📑 Aplicativos utilizados na área": "Fale sobre os aplicativos utilizados na área da saúde com exemplos e benefícios.",

        "📑 Presença da tecnologia no cotidiano": "Análise da presença da tecnologia no cotidiano, com ênfase na geração alfa e no perfil dos novos alunos em relação à tecnologia.",

        "📑 Tecnologias emergentes na Saúde": "Fale sobre a introdução às tecnologias emergentes na saúde.",

        "📑 Aplicabilidade das tecnologias emergentes": "Aplicabilidade das tecnologias emergentes na área da saúde, destacando os seguintes temas: Inteligência artificial (IA) - Realidade aumentada e virtual - Robótica - Internet das coisas (IoT) - Metaversos - Impressora 3D - Big Data - Machine Learning."

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
            st.image(message["image_url"], caption="Exemplo visual encontrado no Google")

input_usuario = st.chat_input("Pergunte algo ou peça uma imagem...")
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
                
                img_url = None
                
                # BUSCA DE IMAGEM VIA SERPER.DEV (GOOGLE IMAGES)
                if any(x in prompt_final.lower() for x in ["imagem", "foto", "mostre", "veja", "figura"]):
                    try:
                        serper_key = st.secrets["SERPER_API_KEY"]
                        url_serper = "https://google.serper.dev/images"
                        payload = {"q": f"{prompt_final} health medical technology diagram", "num": 1}
                        headers = {'X-API-KEY': serper_key, 'Content-Type': 'application/json'}
                        
                        response_serper = requests.post(url_serper, headers=headers, json=payload)
                        search_results = response_serper.json()
                        
                        if search_results.get('images'):
                            img_url = search_results['images'][0]['imageUrl']
                            resposta_texto = f"Encontrei uma imagem técnica relacionada a '{prompt_final}':"
                            st.markdown(resposta_texto)
                            st.image(img_url)
                            st.session_state.messages.append({"role": "assistant", "content": resposta_texto, "image_url": img_url})
                    except Exception as img_err:
                        st.warning("Não foi possível carregar a imagem do Google no momento. Vou explicar via texto.")

                # RESPOSTA TEXTUAL (PDF OU DESCRIÇÃO)
                if img_url is None:
                    prompt_template = ChatPromptTemplate.from_template(
                        "Responda em PT-BR de forma clara e acadêmica. Se o usuário pediu uma imagem e houve falha na busca, "
                        "descreva detalhadamente o que ele veria em um diagrama sobre o assunto. Use o contexto: {context}\nPergunta: {input}"
                    )
                    chain = create_retrieval_chain(base.as_retriever(), create_stuff_documents_chain(llm, prompt_template))
                    response = chain.invoke({"input": prompt_final})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                
                if len(st.session_state.messages) <= 2: st.rerun()
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
