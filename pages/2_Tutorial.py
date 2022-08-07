import streamlit as st

st.set_page_config(
    page_title="Rules extraction", page_icon="📐", initial_sidebar_state="expanded",
)

css = """
<style>
    .main .block-container {padding-top: 3rem;}
    div[data-testid=stSidebarNav] > ul:nth-of-type(1) {padding-top: 5rem;}
    footer {visibility: hidden;}
    ul[data-testid=main-menu-list] > ul:nth-of-type(5) > li:nth-of-type(1) {display: none;}
    ul[data-testid=main-menu-list] > div:nth-of-type(3) {display: none;}
    .css-xjsf0x.e10mrw3y1 {display: none;}
</style>"""
st.markdown(css, unsafe_allow_html=True)

st.markdown("## Work in progress... :construction:")

with st.sidebar:
    st.subheader(":bug: Report any issue or suggestion:" )
    st.markdown('[**Go to Github**](https://github.com/santiagohy/grammar-rules-extraction/issues/new)')
    st.subheader("🔗 Useful websites:" )
    col1, col2 = st.columns(2)
    col1.markdown('[**Grew-match**](http://match.grew.fr/)')
    col2.markdown('[**Universal tables**](http://tables.grew.fr/)')
    col1, col2 = st.columns(2)
    col1.markdown('[**UD Guidelines**](https://universaldependencies.org/guidelines.html)')
    col2.markdown('[**SUD Guidelines**](https://surfacesyntacticud.github.io/guidelines/u/)')
    col1, col2 = st.columns(2)
    col1.markdown('[**UD Corpora**](https://universaldependencies.org/#download)')
    col2.markdown('[**SUD Corpora**](https://surfacesyntacticud.github.io/data/)')
