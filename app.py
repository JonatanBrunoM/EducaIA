import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importações para a lógica da Chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate  # <--- ESSA LINHA AQUI

# 1. Configuração da Página
st.set_page_config(page_title="EducaIA | Tutor Inteligente", page_icon="🤖", layout="wide")

# Estilização para esconder elementos desnecessários e melhorar o visual
st.markdown("""
    <style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {background-color: #f0f2f6;}
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA BIBLIOTECA (BANCO DE DADOS) ---
# Adicione aqui os nomes de todos os PDFs que você subir no GitHub
LIVROS = ["ebook1.pdf", "ebook2.pdf", "ebook3.pdf", "ebook4.pdf"]

@st.cache_resource
def processar_base():
    documentos = []
    for arquivo in LIVROS:
        if os.path.exists(arquivo):
            loader = PyPDFLoader(arquivo)
            documentos.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    textos = text_splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(textos, embeddings)

# --- INTERFACE DA BARRA LATERAL ---
with st.sidebar:
    st.title("🤖 EducaIA")
    st.markdown("O assistente inteligente da nossa disciplina.")
    st.markdown("---")
    
    st.subheader("💡 Sugestões de Pesquisa")
    st.info("Clique em um tema para iniciar a consulta automática:")

    # --- LOCAL PARA VOCÊ ADICIONAR/ALTERAR AS SUGESTÕES ---
    # Basta copiar o bloco de 'if st.button' para criar novas sugestões
    
    if st.button("Quais são as evoluções das tencologias?"):
        st.session_state.sugestao_clicada = "Apresentação da evolução das tecnologias digitais de informação e comunicação na gestão em saúde."

    if st.button("📑 Conceitos Fundamentais"):
        st.session_state.sugestao_clicada = "Quais são os conceitos fundamentais apresentados no material?"

    if st.button("🧪 Explicação de Fórmulas"):
        st.session_state.sugestao_clicada = "Explique as principais fórmulas ou metodologias citadas nos textos."

    if st.button("📝 Simulado de Prova"):
        st.session_state.sugestao_clicada = "Crie 3 questões de múltipla escolha com base no conteúdo para eu treinar."

    # -------------------------------------------------------
    
    st.markdown("---")
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

# --- MOTOR DE INTELIGÊNCIA ---
if any(os.path.exists(f) for f in LIVROS):
    base = processar_base()
else:
    st.error("Erro crítico: Banco de dados não localizado.")
    st.stop()

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Lógica para capturar o clique da sugestão
prompt_final = None
if "sugestao_clicada" in st.session_state and st.session_state.sugestao_clicada:
    prompt_final = st.session_state.sugestao_clicada
    st.session_state.sugestao_clicada = None # Limpa para não repetir

# Interface de Chat Principal
st.title("📚 Central de Conhecimento")
st.caption("Consulte o banco de dados da disciplina através de IA")

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura entrada do usuário (pelo campo de texto)
input_usuario = st.chat_input("Digite sua dúvida aqui...")

# Se o usuário digitou algo ou clicou em uma sugestão
if input_usuario or prompt_final:
    texto_da_pergunta = input_usuario if input_usuario else prompt_final
    
    st.session_state.messages.append({"role": "user", "content": texto_da_pergunta})
    with st.chat_message("user"):
        st.markdown(texto_da_pergunta)

    try:
        chave = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(groq_api_key=chave, model_name="llama-3.1-8b-instant", temperature=0.3)
        
        prompt_template = ChatPromptTemplate.from_template("""
        Você é o EducaIA, um tutor acadêmico especializado e prestativo.
        Sua resposta deve ser baseada ESTRITAMENTE no contexto fornecido abaixo.
        Se a resposta não estiver no contexto, diga educadamente que o material disponível não aborda esse ponto específico.
        
        Responda sempre em Português do Brasil (PT-BR), usando uma linguagem clara e formatando com tópicos quando necessário.
        
        Contexto: {context}
        Pergunta: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(base.as_retriever(), document_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Consultando banco de dados..."):
                response = retrieval_chain.invoke({"input": texto_da_pergunta})
                texto_resposta = response["answer"]
                st.markdown(texto_resposta)
        
        st.session_state.messages.append({"role": "assistant", "content": texto_resposta})
    except Exception as e:
        st.error(f"Erro na consulta: {e}")
    else:
    st.error("ERRO: Os PDFs não foram encontrados. Verifique se os nomes no GitHub estão como livro1.pdf, etc.")
