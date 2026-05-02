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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sugestao_clicada" not in st.session_state:
    st.session_state.sugestao_clicada = None
if "quiz_atual" not in st.session_state:
    st.session_state.quiz_atual = None

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown(f'<div class="sidebar-header"><img src="data:image/png;base64,{bin_str_mini}" class="sidebar-logo"><h1 style="font-size: 22px; margin: 0;">EducaIA</h1></div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; opacity: 0.7; margin-bottom: 0;'>Assistente Acadêmico Digital</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-top-button">', unsafe_allow_html=True)
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.session_state.quiz_atual = None
        st.rerun()
    
    # FUNCIONALIDADE 1: BOTÃO DE QUIZ INTERATIVO
    if st.button("🧠 Gerar Quiz Interativo"):
        st.session_state.quiz_atual = None # Limpa quiz anterior
        st.session_state.sugestao_clicada = "Gere uma questão de múltipla escolha baseada nos PDFs. Use EXATAMENTE este formato: PERGUNTA: [texto] | A) [opção] | B) [opção] | C) [opção] | CORRETA: [letra]"
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

if not st.session_state.messages:
    st.markdown(f'<div class="welcome-text"><h1 class="welcome-title">Olá! Eu sou o EducaIA</h1><p style="font-size: 20px; opacity: 0.8;">Vamos transformar seus PDFs em conhecimento?</p></div>', unsafe_allow_html=True)

# Exibição do Histórico
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
                llm = ChatGroq(groq_api_key=chave_groq, model_name="llama-3.1-8b-instant", temperature=0.4)
                
                img_urls_list = []
                
                # 1. Lógica de Imagem
                if any(x in prompt_final.lower() for x in ["imagem", "foto", "mostre", "veja", "figura"]):
                    try:
                        serper_key = st.secrets["SERPER_API_KEY"]
                        url_serper = "https://google.serper.dev/images"
                        payload = {"q": prompt_final, "num": 3}
                        headers = {'X-API-KEY': serper_key, 'Content-Type': 'application/json'}
                        res = requests.post(url_serper, headers=headers, json=payload)
                        search_results = res.json()
                        if search_results.get('images'):
                            img_urls_list = [img['imageUrl'] for img in search_results['images']]
                            st.markdown(f"Imagens encontradas para: {prompt_final}")
                            cols = st.columns(len(img_urls_list))
                            for idx, url in enumerate(img_urls_list): cols[idx].image(url)
                            st.session_state.messages.append({"role": "assistant", "content": f"Galeria sobre {prompt_final}", "image_url": img_urls_list})
                    except: pass

                # 2. Lógica de Texto / Quiz Interativo
                if not img_urls_list:
                    is_quiz = "múltipla escolha" in prompt_final.lower() or "flashcard" in prompt_final.lower()
                    
                    prompt_template = ChatPromptTemplate.from_template(
                        "Você é um tutor acadêmico. Responda em PT-BR usando o contexto: {context}\n"
                        "Pergunta: {input}"
                    )
                    
                    chain = create_retrieval_chain(base.as_retriever(), create_stuff_documents_chain(llm, prompt_template))
                    response = chain.invoke({"input": prompt_final})
                    full_text = response["answer"]
                    
                    # Processamento do Quiz Interativo
                    if is_quiz and "|" in full_text:
                        try:
                            # Divide a resposta da IA nos componentes do quiz
                            partes = full_text.split("|")
                            pergunta = partes[0].replace("PERGUNTA:", "").strip()
                            opcoes = [partes[1].strip(), partes[2].strip(), partes[3].strip()]
                            correta = partes[4].replace("CORRETA:", "").strip().upper()
                            
                            # Salva no estado para o Streamlit não perder no clique do radio
                            st.session_state.quiz_atual = {"p": pergunta, "o": opcoes, "c": correta}
                        except:
                            st.markdown(full_text) # Fallback se a IA errar o formato
                    else:
                        st.markdown(full_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_text})

                # Renderização da interface de resposta do Quiz (se houver um quiz ativo)
                if st.session_state.quiz_atual:
                    q = st.session_state.quiz_atual
                    st.markdown(f"### 📝 Desafio: {q['p']}")
                    
                    # Cria os botões de opção
                    escolha = st.radio("Selecione a alternativa correta:", q['o'], index=None, key="quiz_radio")
                    
                    if escolha:
                        # Verifica se a letra inicial da escolha (A, B ou C) bate com a CORRETA
                        letra_usuario = escolha[0].upper()
                        if letra_usuario == q['c']:
                            st.success(f"🎯 Excelente! A alternativa {q['c']} está correta.")
                        else:
                            st.error(f"❌ Não foi dessa vez. Você marcou {letra_usuario}, mas a correta é a {q['c']}.")
                        
                        # Limpa o quiz após responder para não ficar travado na tela
                        if st.button("Próxima pergunta"):
                            st.session_state.quiz_atual = None
                            st.rerun()

                if len(st.session_state.messages) <= 2: st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")
