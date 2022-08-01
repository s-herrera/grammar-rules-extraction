import streamlit as st

st.set_page_config(
    page_title="Rules extraction", page_icon="📐", initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            ul[data-testid=main-menu-list] > ul:nth-of-type(4) > li:nth-of-type(1) {display: none;}
            ul[data-testid=main-menu-list] > div:nth-of-type(2) {display: none;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.markdown("## Work in progress... :construction:")

