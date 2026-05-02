# --- NO SET_PAGE_CONFIG ---
# Dica: Se o favicon continuar pequeno, tente recortar a imagem 'logomini.png' 
# removendo espaços em branco nas bordas antes de subir no GitHub.
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
    
    /* Restante do seu CSS anterior... */
    </style>
""", unsafe_allow_html=True)
