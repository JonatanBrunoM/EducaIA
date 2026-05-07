import streamlit as st
import os
import io
import base64
import requests
import random
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from google_auth_oauthlib.flow import Flow

if "connected" not in st.session_state:
    st.session_state.connected = False

# 1. Configuração da Página
st.set_page_config(
    page_title="EducaIA", 
    page_icon="logomini.png", 
    layout="wide"
)

# --- CONFIGURAÇÃO LOGIN GOOGLE ---
client_config = {
    "web": {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [st.secrets["GOOGLE_REDIRECT_URI"]],
    }
}

# Processar o retorno do Google (Callback)
query_params = st.query_params
if "code" in query_params and not st.session_state.get('connected'):
    try:
        flow = Flow.from_client_config(
            client_config,
            scopes=['https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email', 'openid'],
            redirect_uri=st.secrets["GOOGLE_REDIRECT_URI"]
        )
        flow.fetch_token(code=query_params["code"])
        credentials = flow.credentials
        user_info_service = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        ).json()
        
        st.session_state.connected = True
        st.session_state.name = user_info_service.get("name")
        st.session_state.email = user_info_service.get("email")
        st.session_state.picture = user_info_service.get("picture")
        
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Erro ao processar login: {e}")
        st.query_params.clear()

# Definição de Variáveis de Usuário (Modo Híbrido: Logado ou Visitante)
if st.session_state.get('connected'):
    user_info = {
        "name": st.session_state.get('name'),
        "email": st.session_state.get('email'),
        "picture": st.session_state.get('picture'),
        "given_name": st.session_state.get('name').split()[0] if st.session_state.get('name') else "Estudante"
    }
else:
    user_info = {
        "name": "Visitante",
        "email": "Sem login",
        "picture": "https://www.gstatic.com/images/branding/product/1x/avatar_circle_blue_512dp.png",
        "given_name": "Estudante"
    }

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
    pdf.cell(0, 10, txt="EducaIA - Resumo Acadêmico", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    texto_limpo = texto.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, txt=texto_limpo)
    pdf_bytes = pdf.output(dest='S')
    return bytes(pdf_bytes) if isinstance(pdf_bytes, bytearray) else pdf_bytes

# 2. CSS Customizado Atualizado
st.markdown(f"""
    <style>
    /* Estiliza o botão de link nativo (Login Google) */
    .stElementContainer div[data-testid="stLinkButton"] a {{
        background-color: #1e86c8 !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        text-align: center !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 10px 20px !important;
        transition: background-color 0.3s ease !important;
    }}
    
    .stElementContainer div[data-testid="stLinkButton"] a:hover {{
        background-color: #156b9f !important;
        color: white !important;
        text-decoration: none !important;
    }}

    /* Estilo da Sidebar */
    .sidebar-top-button {{
        padding-top: 10px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
    }}
    
    .faculdade-logo {{
        position: absolute; 
        top: -55px; 
        left: 50px; 
        width: 150px; 
        z-index: 99;
    }}
    
    .sidebar-header {{ 
        display: flex; 
        align-items: center; 
        gap: 12px; 
        margin-bottom: 10px; 
    }}
    
    .sidebar-logo {{ 
        width: 35px; 
        height: auto; 
    }}
    
    /* Botões normais da Sidebar */
    .stButton > button {{
        border-radius: 20px; 
        border: 1px solid #444746; 
        width: 100%; 
        text-align: left; 
        padding: 10px 20px;
    }}
    
    /* Botões de sugestão (estilo itálico azul) */
    .suggestion-btn button {{
        background-color: transparent !important;
        border: 1px solid #1e86c8 !important;
        color: #1e86c8 !important;
        font-style: italic !important;
        font-size: 13px !important;
        height: auto !important;
        text-align: center !important;
    }}
    
    /* Esconder elementos desnecessários */
    .stDeployButton {{display:none;}}
    footer {{visibility: hidden;}}
    
    /* Texto de Boas-vindas */
    .welcome-text {{ 
        text-align: center; 
        margin-top: 10vh; 
    }}
    
    .welcome-title {{
        font-size: 50px; 
        font-weight: 600; 
        background: linear-gradient(90deg, #1e86c8, #8ac5e2);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- MOTOR DE IA ---
# 1. Defina a lista mestre primeiro
ARQUIVOS_DISPONIVEIS = ["ebook1.pdf", "ebook2.pdf", "ebook3.pdf", "ebook4.pdf", "datacenter.pdf", "internetdascoisas.pdf"]

with st.sidebar:
    st.subheader("📚 Fonte de Conhecimento")
    # 2. Use a lista mestre nas opções
    livros_selecionados = st.multiselect(
        "Selecione os arquivos para basear o estudo:",
        options=ARQUIVOS_DISPONIVEIS,
        default=ARQUIVOS_DISPONIVEIS 
    )
    # 3. Atualize a variável LIVROS que será usada na base
    LIVROS = livros_selecionados

@st.cache_resource
def processar_base(lista_arquivos):
    documentos = []
    for arquivo in lista_arquivos:
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
if "ultimo_resumo" not in st.session_state:
    st.session_state.ultimo_resumo = None
if "proximas_perguntas" not in st.session_state:
    st.session_state.proximas_perguntas = []

# --- BARRA LATERAL ---
with st.sidebar:
    if st.session_state.get('connected'):
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <img src="{user_info['picture']}" style="width: 40px; border-radius: 50%;">
                <div>
                    <p style="margin: 0; font-weight: bold; font-size: 14px;">{user_info['name']}</p>
                    <p style="margin: 0; font-size: 12px; opacity: 0.7;">{user_info['email']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        st.info("Entre para personalizar sua experiência.")
        auth_url = (
            f"https://accounts.google.com/o/oauth2/auth?"
            f"response_type=code&client_id={st.secrets['GOOGLE_CLIENT_ID']}&"
            f"redirect_uri={st.secrets['GOOGLE_REDIRECT_URI']}&"
            f"scope=https://www.googleapis.com/auth/userinfo.profile%20https://www.googleapis.com/auth/userinfo.email%20openid&prompt=select_account"
        )
        st.link_button("Efetuar login", auth_url, use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(f'<div class="sidebar-header"><img src="data:image/png;base64,{bin_str_mini}" class="sidebar-logo"><h1 style="font-size: 22px; margin: 0;">EducaIA</h1></div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; opacity: 0.7; margin-bottom: 0;'>Assistente Acadêmico Digital</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🎯 Modo de Estudo")
    modo_estudo = st.radio(
        "Como prefere que eu responda?",
        ["🎓 Tutor (Didático)", "📝 Resumo (Direto)", "🔬 Científico (Técnico)"],
        index=0
    )
    
    st.markdown('<div class="sidebar-top-button">', unsafe_allow_html=True)
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.session_state.quiz_atual = None
        st.session_state.ultimo_resumo = None
        st.session_state.proximas_perguntas = []
        st.rerun()
    
    if st.button("🧠 Gerar Quiz"):
        st.session_state.quiz_atual = None # Limpa o anterior
        st.session_state.sugestao_clicada = (
            "Gere 10 questões de múltipla escolha aleatórias e diversificadas, "
            "escolhendo diferentes tópicos misturados dos PDFs selecionados. "
            "Não foque apenas em um capítulo. "
            "Use EXATAMENTE este formato para CADA uma: "
            "PERGUNTA: [texto] | A) [op1] | B) [op2] | C) [op3] | D) [op4] | CORRETA: [letra]"
        )
    
    if st.button("📄 Gerar Resumo da Conversa"):
        if len(st.session_state.messages) > 0:
            conteudo_chat = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages if "image_url" not in m])
            st.session_state.sugestao_clicada = f"Com base exclusivamente na nossa conversa abaixo, crie um resumo estruturado para meus estudos:\n\n{conteudo_chat}"
        else:
            st.warning("Inicie uma conversa primeiro!")
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
        "📑 Presença da tecnologia no cotidiano": "Análise da presença da tecnologia no cotidiano, com ênfase na geração alfa e no perfil dos novos alunos.",
        "📑 Tecnologias emergentes na Saúde": "Fale sobre a introdução às tecnologias emergentes na saúde.",
        "📑 Aplicabilidade das tecnologias": "Aplicabilidade das tecnologias emergentes na área da saúde (IA, IoT, Big Data, etc)."
    }
    
    for label, prompt in sugestoes.items():
        if st.button(label): 
            st.session_state.sugestao_clicada = prompt

    st.markdown("---")
    st.subheader("🔍 Glossário Acadêmico")
    termos = {
        "Cibercultura": "Explique o concept de Cibercultura conforme os documentos.",
        "IA na Saúde": "O que é Inteligência Artificial aplicada à saúde?",
        "Geração Alfa": "Quem é a Geração Alfa no contexto educacional tecnológico?",
        "IoT (Internet das Coisas)": "O que significa IoT e como se aplica à saúde?",
        "Big Data": "Explique o conceito de Big Data no setor de saúde."
    }
    for termo, prompt_termo in termos.items():
        if st.button(f"🔍 {termo}"):
            st.session_state.sugestao_clicada = prompt_termo

    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("⚙️"):
        st.caption("Materiais na base:")
        for arquivo in LIVROS:
            status = "✅" if os.path.exists(arquivo) else "❌"
            st.markdown(f"**{status}** `{arquivo}`")

# --- ÁREA PRINCIPAL ---
st.markdown(f'<img src="data:image/png;base64,{bin_str_faculdade}" class="faculdade-logo">', unsafe_allow_html=True)
base = processar_base(LIVROS)

AVATAR_USER = user_info['picture'] 
AVATAR_AI = f"data:image/png;base64,{bin_str_mini}"

# 1. Criação das Abas
tab_aula, tab_quiz = st.tabs(["📖 Aula Interativa", "📝 Espaço de Desafios"])

# --- CONTEÚDO DA ABA AULA ---
with tab_aula:
    if not st.session_state.messages:
        st.markdown(f'<div class="welcome-text"><h1 class="welcome-title">Olá, {user_info["given_name"]}!</h1><p style="font-size: 20px; opacity: 0.8;">Eu sou o EducaIA. Vamos estudar sobre qual assunto hoje?</p></div>', unsafe_allow_html=True)

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

    if st.session_state.proximas_perguntas and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.write("---")
        st.caption("Sugestão de continuação:")
        cols_sug = st.columns(len(st.session_state.proximas_perguntas))
        for i, sug in enumerate(st.session_state.proximas_perguntas):
            with cols_sug[i]:
                st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
                if st.button(sug, key=f"btn_sug_{i}"):
                    st.session_state.sugestao_clicada = sug
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

# --- CONTEÚDO DA ABA QUIZ ---
with tab_quiz:
    if st.session_state.quiz_atual:
        # Se for uma lista de questões
        for idx, q in enumerate(st.session_state.quiz_atual):
            with st.expander(f"Questão {idx+1}", expanded=(idx==0)):
                with st.form(key=f"form_quiz_{idx}"):
                    st.write(q['p'])
                    escolha = st.radio("Opções:", q['o'], key=f"rad_{idx}")
                    if st.form_submit_button("Confirmar Resposta"):
                        if escolha:
                            # Pega a primeira letra da escolha (ex: "A")
                            letra_escolhida = escolha[0].upper()
        
                            if letra_escolhida == q['c']:
                                st.success(f"🎯 Correto! A alternativa é a {q['c']}.")
                            else:
                                # Aqui mostramos qual era a resposta correta
                                st.error(f"❌ Incorreto.")
                                st.info(f"💡 A resposta correta era a alternativa: **{q['c']}**")
                        else:
                            st.warning("Por favor, selecione uma opção.")
        
        if st.button("Limpar Quiz"):
            st.session_state.quiz_atual = None
            st.rerun()
    else:
        st.write("Nenhum quiz ativo no momento. Peça um quiz na aba de Aula ou use o botão na barra lateral!")

# --- INPUT E LÓGICA DE IA ---
input_usuario = st.chat_input("Pergunte algo...")
prompt_final = input_usuario if input_usuario else st.session_state.sugestao_clicada

if prompt_final:
    st.session_state.sugestao_clicada = None 
    st.session_state.messages.append({"role": "user", "content": prompt_final})
    
    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Processando..."):
            try:
                chave_groq = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(groq_api_key=chave_groq, model_name="llama-3.1-8b-instant", temperature=0.4)
                
                instrucao_tom = {
                    "🎓 Tutor (Didático)": "Use linguagem simples, didática e exemplos claros.",
                    "📝 Resumo (Direto)": "Seja muito breve, use tópicos e foque nos pontos principais.",
                    "🔬 Científico (Técnico)": "Use termos técnicos avançados e linguagem acadêmica formal."
                }
                tom_selecionado = instrucao_tom[modo_estudo]

                full_text = "" # Inicializa a variável para evitar erro de referência
                img_urls_list = []
                
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
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"Aqui estão algumas imagens sobre: {prompt_final}", 
                                "image_url": img_urls_list
                            })
                            st.rerun() 
                    except Exception as e:
                        pass

                if not img_urls_list:
                    if "nossa conversa abaixo" in prompt_final:
                        full_text = llm.invoke(f"{tom_selecionado}\n\n{prompt_final}").content
                        st.session_state.ultimo_resumo = full_text
                        st.session_state.proximas_perguntas = []
                    else:
                        template_texto = (
                            "Você é um tutor acadêmico em PT-BR. " + tom_selecionado + "\n"
                            "Responda usando o contexto: {context}\n"
                            "Pergunta: {input}\n\n"
                            "IMPORTANTE: Ao final da resposta, adicione sempre uma linha começando exatamente com 'SUGESTÃO:' "
                            "e liste 1 pergunta curta para o aluno continuar estudando este tema."
                            "REGRAS DE RESPOSTA:\n"
                            "1. Se o aluno pedir um QUIZ ou questão, use EXATAMENTE: PERGUNTA: [texto] | A) [op1] | B) [op2] | C) [op3] | D) [op4] | CORRETA: [Letra]\n"
                            "2. Caso contrário, responda normalmente e termine com 'SUGESTÃO: [pergunta curta]'."
                        )
                        
                        prompt_template = ChatPromptTemplate.from_template(template_texto)
                        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
                        if base is None:
                            st.error("Por favor, selecione ao menos um arquivo na barra lateral para começar.")
                            st.stop()
                        chain = create_retrieval_chain(base.as_retriever(), combine_docs_chain)
                        
                        response = chain.invoke({"input": prompt_final})
                        raw_answer = response["answer"]
                        
                        if "SUGESTÃO:" in raw_answer:
                            partes = raw_answer.split("SUGESTÃO:")
                            full_text = partes[0].strip()
                            sugestao_unica = partes[1].strip().split('\n')[0] 
                            st.session_state.proximas_perguntas = [sugestao_unica]
                        else:
                            full_text = raw_answer
                            st.session_state.proximas_perguntas = []

                    if "PERGUNTA:" in full_text:
                        try:
                            linhas_questoes = full_text.strip().split('\n')
                            lista_quizzes = []
        
                            for linha in linhas_questoes:
                                if "|" in linha and "PERGUNTA:" in linha:
                                    partes = linha.split("|")
                                    pergunta = partes[0].replace("PERGUNTA:", "").strip()
                                    opcoes = [partes[1].strip(), partes[2].strip(), partes[3].strip(), partes[4].strip()]
                                    correta = partes[5].replace("CORRETA:", "").strip().upper()
                                    correta = correta.replace(")", "").strip()
                                    
                                    lista_quizzes.append({"p": pergunta, "o": opcoes, "c": correta})
        
                            if lista_quizzes:
                                    st.session_state.quiz_atual = lista_quizzes # Agora guarda a lista de 10
                                    random.shuffle(lista_quizzes) # EMBARALHA a ordem das 10 questões
                                    full_text = f"🎯 Preparei {len(lista_quizzes)} desafios para você! Vá até a aba **'📝 Espaço de Desafios'**."
                        except:
                            pass
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_text})
                    st.rerun()

            except Exception as e:
                st.error(f"Ocorreu um erro no motor de IA: {e}")

# Botão de Download PDF
if st.session_state.ultimo_resumo:
    st.divider()
    pdf_data = gerar_pdf_resumo(st.session_state.ultimo_resumo)
    st.download_button(
        label="📥 Baixar Resumo em PDF",
        data=pdf_data,
        file_name="resumo_educaia.pdf",
        mime="application/pdf",
        use_container_width=True
    )
