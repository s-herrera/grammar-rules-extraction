import streamlit as st

st.set_page_config(page_title="Rules extraction", page_icon="📐", initial_sidebar_state="expanded")

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


@st.experimental_memo(show_spinner=False, suppress_st_warning=True)
def load_files(files, _uploaded_files: list):
    """
    Read several files into a string.
    """
    with st.spinner('Loading treebank...'):
        files = []
        for uploaded_file in _uploaded_files:
            files.append(uploaded_file.getvalue().decode("utf-8"))
        return "".join(files)


st.header("Tools 🧰")

with st.sidebar:
    st.subheader(":bug: Report any issue or suggestion:")
    st.markdown('[**Go to Github**](https://github.com/santiagohy/grammar-rules-extraction/issues/new)')
    st.subheader("🔗 Useful websites:")
    col1, col2 = st.columns(2)
    col1.markdown('[**Grew-match**](http://match.grew.fr/)')
    col2.markdown('[**Universal tables**](http://tables.grew.fr/)')
    col1, col2 = st.columns(2)
    col1.markdown('[**UD Guidelines**](https://universaldependencies.org/guidelines.html)')
    col2.markdown('[**SUD Guidelines**](https://surfacesyntacticud.github.io/guidelines/u/)')
    col1, col2 = st.columns(2)
    col1.markdown('[**UD Corpora**](https://universaldependencies.org/#download)')
    col2.markdown('[**SUD Corpora**](https://surfacesyntacticud.github.io/data/)')

st.markdown("Some tools to better explore treebanks")
st.markdown("- [ Concatenate files](#concatenate-files)")

st.markdown("#### Concatenate files")
with st.form("form4"):
    uploaded_files = uploaded_files = st.file_uploader("CoNLL/CoNLL-U accepted", type=[".conll", ".conllu"], accept_multiple_files=True)
    filenames = [f.name for f in uploaded_files]
    col1, col2 = st.columns([4, 1])
    inpt1 = col1.text_input("File name")
    inpt2 = col2.text_input("Extension", value="conllu")
    submitted4 = st.form_submit_button("Upload")

    if uploaded_files:
        if inpt1 and inpt2:
            files = {i: (uploaded_files[i].name, uploaded_files[i].size) for i in range(len(uploaded_files))}
            concat_file = load_files(files, uploaded_files)

if uploaded_files and inpt1 and inpt2:
    _, col2, _ = st.columns([2, 1, 2])
    col2.download_button(label="Download", data=concat_file, file_name=f"{inpt1}.{inpt2}", mime='text')