import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from grewpy import grew, Corpus

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import numpy as np

import tempfile
from typing import Tuple, Dict
from collections import namedtuple

import extraction_tools as et
from texts import CHECKBOX_HELP, P1P2_HELP, P3_HELP, OPTION_HELP, ABOUT

# -----------------------------------------


# @st.experimental_memo(show_spinner=False, suppress_st_warning=True)
# Variable 'files_' is used to check / st.state_session['uploaded_files']
# in cache if the uploaded files have changed
def load_corpora(uploaded_files: list) -> Tuple[Dict, int, int, int]:
    """
    Load corpus in a dictionary and by using Grew.

    Return corpora and its number of sentences and tokens.
    """
    with st.spinner('Loading treebank...'):
        with tempfile.NamedTemporaryFile(mode="wt", encoding="utf-8") as temp:
            for uploaded_file in uploaded_files:
                f = uploaded_file.getvalue().decode("utf-8")
                temp.write(f)
            temp.seek(0)
            corpus = Corpus(temp.name)
            treebank, features = et.conllu_to_dict(temp.name)
            sentences, tokens = et.get_corpus_info(treebank)

    return treebank, corpus, sentences, tokens, features


def convert_df(df):
    """
    Convert a dataframe into a tsv and a json.
    """
    jsn = df.to_json(orient="split", index=False, indent=4)
    tsv = df.to_csv(index=False, encoding="utf-8", sep="\t")
    return jsn, tsv


def get_dataframe(lst: list) -> pd.DataFrame:
    df = pd.DataFrame(lst, columns=["Pattern", "Significance", "Effect", "% of P1&P2", "% of P1&P3"])
    df['Significance'] = df['Significance'].apply(lambda x: et.format_significance(x))
    df = df.sort_values('Significance', ascending=False)
    return df


def get_aggrid_and_response(df: pd.DataFrame) -> Dict:
    df = df.replace(float('inf'), 'Infinity')
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection("single", use_checkbox=True)
    gb.configure_column(field="Significance", type=["numericColumn", "numberColumnFilter"])
    gb.configure_column(field="Effect", type=["numericColumn", "numberColumnFilter",
                                        "customNumericFormat"], precision=3)
    gb.configure_columns(column_names=["% of P1&P2", "% of P1&P3"], type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
    go = gb.build()
    grid_response = AgGrid(
        df,
        gridOptions=go,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        conversion_errors="ignore",
        try_to_convert_back_to_original_types=False,
        enable_enterprise_modules=False)
    return grid_response


def initialize_session_keys(keys: list):
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = ""


def reinitialize_session_keys(keys: list):
    for k in keys:
        st.session_state[k] = ""


def clear_data(keys: list):
    st.experimental_memo.clear()
    reinitialize_session_keys(keys)

# -----------------------------------------


st.set_page_config(
    page_title="Rules extraction",
    page_icon="📐",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={"Get Help": None, "Report a Bug": "https://github.com/santiagohy/grammar-rules-extraction/issues/new"}
)

css = """
<style>
    .main .block-container {padding-top: 1rem;}
    div[data-testid=stSidebarNav] > ul:nth-of-type(1) {padding-top: 4rem;}
    footer {visibility: hidden;}
    ul[data-testid=main-menu-list] > ul:nth-of-type(5) > li:nth-of-type(1) {display: none;}
    ul[data-testid=main-menu-list] > div:nth-of-type(3) {display: none;}
    .css-xjsf0x.e10mrw3y1 {display: none;}
</style>"""
st.markdown(css, unsafe_allow_html=True)

st.markdown("# Rule extraction 📐")

# grew.init()
grew.set_config("sud")

keys = ['uploaded_files', 'filenames', 'pattern1', 'pattern2',
        'pattern3', 'result', 'filenames', 'M', 'n', 'files',
        'sentences', 'tokens', 'treebanks', 'corpus',
        'p1_key', 'p2_key', 'p3_key', 'df']

initialize_session_keys(keys)

with st.expander('🤏 About the app '):
    st.markdown(ABOUT)

# Form 1

with st.form("form1", clear_on_submit=True):

    uploaded_files = st.file_uploader("CoNLL/CoNLL-U accepted", type=[".conll", ".conllu"], accept_multiple_files=True)
    submitted1 = st.form_submit_button('Upload')

    if submitted1:

        reinitialize_session_keys(['uploaded_files', 'filenames', 'files', 'pattern1', 'pattern2', 'pattern3','result'])
        st.session_state['uploaded_files'] = uploaded_files

        if st.session_state['uploaded_files']:
            st.session_state['filenames'] = [f.name for f in uploaded_files]
            st.session_state['files'] = {i: (uploaded_files[i].name, uploaded_files[i].size) for i in range(len(uploaded_files))}
            st.session_state['df'] = pd.DataFrame(np.zeros((3, 3), dtype=int), columns=["p3", "¬ p3", "total"], index=["p2", "¬ p2", "total"])

            st.session_state['treebank'], corpus, sentences, tokens, features = load_corpora(st.session_state['uploaded_files'])
            st.session_state['features'] = features
            st.session_state['corpus'] = corpus
            st.session_state['sentences'] = sentences
            st.session_state['tokens'] = tokens

if st.session_state['uploaded_files']:

    # Side bar
    if len(st.session_state['filenames']) <= 3:
        st.sidebar.json(st.session_state['filenames'])
    else:
        st.sidebar.json(st.session_state['filenames'], expanded=False)

    with st.sidebar:
        col1, col2, col3 = st.columns([0.5, 1, 1], gap="small")  # 0.3,0.9,1
        col1.metric("# Files", len(st.session_state['filenames']))
        col2.metric("# Sentences", st.session_state['sentences'])
        col3.metric("# Tokens", st.session_state['tokens'], help="It includes multiword tokens (e.g. n-n+1 indexed tokens)")

    st.sidebar.markdown("#### Nodes features:")
    featsholder = st.sidebar.empty()

    nresholder = st.sidebar.empty()
    p3holder = st.sidebar.empty()
    tableholder = st.sidebar.empty()
    linkholder = st.sidebar.empty()
    buttonsholder = st.sidebar.empty()

    nresholder.markdown("#### No. of results:")
    p3holder.markdown("#### Selection:")
    tableholder.table(st.session_state['df'])

    # Form 2
    with st.form(key="form2"):

        st.subheader("First two queries")
        col1, col2 = st.columns(2)
        col1.text_area("Pattern 1", value="", key="p1_key", height=80)
        col2.text_area("Pattern 2", value="", key="p2_key", height=80, help=P1P2_HELP)

        p1matched = col1.text("")
        p2matched = col2.text("")

        submitted2 = st.form_submit_button("Match")

        if submitted2:

            p1 = st.session_state['p1_key']
            p2 = st.session_state['p2_key']

            for p in (p1, p2):
                validation = et.is_valid_pattern(p)
                if isinstance(validation, Exception):
                    st.exception(validation)
                    st.stop()

            if p1 and p2:

                reinitialize_session_keys(keys=['pattern3', 'result'])
                st.session_state['pattern1'] = p1
                st.session_state['pattern2'] = p2

                # st.session_state['p1grew'] = et.build_GrewPattern(p1)
                # st.session_state['p1grew'] = p1
                # st.session_state['p2grew'] = et.build_GrewPattern(p2)

                try:
                    matchs, allfeatures = et.get_patterns_info(st.session_state['corpus'], st.session_state['treebank'], st.session_state['pattern1'])
                except grew.GrewError as e:
                    st.exception(grew.GrewError(e.value))
                    st.stop()

                try:
                    M, n = et.compute_fixed_totals(matchs, st.session_state['pattern1'], st.session_state['pattern2'], st.session_state['corpus'])
                except grew.GrewError as e:
                    st.exception(grew.GrewError(e.value))
                    st.stop()

                st.session_state["M"] = M
                st.session_state["n"] = n
                st.session_state["matchs"] = matchs
                st.session_state["allfeatures"] = allfeatures
                st.session_state['df'] = pd.DataFrame(np.zeros((3, 3), dtype=int), columns=["p3", "¬ p3", "total"], index=["p2", "¬ p2", "total"])

            else:
                reinitialize_session_keys(['pattern1', 'pattern2'])
                st.session_state['df'] = pd.DataFrame(np.zeros((3, 3), dtype=int), columns=["p3", "¬ p3", "total"], index=["p2", "¬ p2", "total"])
                tableholder.table(st.session_state['df'])
                st.info("Complete both patterns!")

else:
    st.info("Upload a corpora")
    st.stop()

# Form 3
if st.session_state['pattern1'] and st.session_state['pattern2']:

    p1matched.markdown(f":zap: Matching `{st.session_state['pattern1']}`")
    p2matched.markdown(f":zap: Matching `{st.session_state['pattern2']}`")

    M = st.session_state['M']
    n = st.session_state['n']

    featsholder.json(st.session_state['allfeatures'], expanded=False)

    st.session_state['df']['total'] = [n, M-n, M]
    tableholder.table(st.session_state['df'])

    with st.form(key="form3"):
        st.subheader("Last query")

        st.text_area("Pattern 3 or Key(s) to cluster", value="", key="p3_key", help=P3_HELP, height=80)
        p3matched = st.text("")
        option = st.radio("Combination mode", ('Simple combination', "All possible combinations"), horizontal=True, help=OPTION_HELP)
        if option == 'Simple combination':
            combination_type = False
        else:
            combination_type = True
        submitted3 = st.form_submit_button("Get results")

        if submitted3:

            st.session_state['pattern3'] = st.session_state['p3_key']
            st.session_state['df'][["p3", "¬ p3"]] = np.zeros([3, 2], dtype=int)
            tableholder.table(st.session_state['df'])

            if st.session_state['pattern3']:
                key_predictors = et.get_key_predictors(st.session_state['pattern1'], st.session_state['pattern3'], st.session_state['features'])

                if not key_predictors:
                    validation = et.is_valid_pattern(st.session_state['pattern3'])
                    if isinstance(validation, Exception):
                        st.exception(validation)
                        st.stop()

                patterns = et.get_patterns(st.session_state['treebank'], st.session_state["matchs"], st.session_state['pattern3'], key_predictors, combination_type)

                try:
                    result, tables = et.rules_extraction(st.session_state['corpus'], patterns, st.session_state['pattern1'], st.session_state['pattern2'], st.session_state["M"], st.session_state["n"])
                except grew.GrewError as e:
                    st.exception(grew.GrewError(e.value))
                    st.stop()

                st.session_state['keys'] = bool(key_predictors)
                st.session_state['result'] = [[]]
                if result:
                    st.session_state['result'] = result
                    st.session_state['tables'] = tables
            else:
                st.info("Complete the last pattern!")

# Results
    if st.session_state['pattern3']:

        p3matched.markdown(f":zap: Matching `{st.session_state['pattern3']}`")

        if st.session_state['result'][0]:

            result = st.session_state['result']
            if combination_type:
                checkbox = st.checkbox("Get only the most significant subsets", help=CHECKBOX_HELP)
                if checkbox:
                    subsets = et.get_significant_subsets(result)
                    result = [r for sset in subsets for r in result if sset == tuple(x.strip() for x in r[0].split(";"))]
                df = get_dataframe(result)
                grid_response = get_aggrid_and_response(df)
            else:
                df = get_dataframe(result)
                grid_response = get_aggrid_and_response(df)

            nresholder.markdown(f"#### No. of results: `{len(result)}`")

            if grid_response['selected_rows']:

                # Get contengency table from selection
                pattern3 = grid_response['selected_rows'][0]['Pattern']
                table = st.session_state['tables'][pattern3]
                p3holder.markdown(f"""
                #### Selection:
                    {pattern3}""")
                dfnewvalues = np.append(table, [np.sum(table, axis=0)], axis=0)
                st.session_state['df'][["p3", "¬ p3"]] = dfnewvalues
                tableholder.table(st.session_state['df'])

                # Grew-match link
                if st.session_state['keys']:
                    st.session_state['pattern3'] = f"pattern {{ {pattern3} }}"
                else:
                    st.session_state['pattern3'] = pattern3
                link = et.get_GrewMatch_link(st.session_state['filenames'], st.session_state['pattern1'], st.session_state['pattern2'], st.session_state['pattern3'])
                linkholder.markdown(f'🔗 [**Grew-match link**]({link})')

            else:
                st.session_state['df'][["p3", "¬ p3"]] = st.session_state['df'][["p3", "¬ p3"]] = np.zeros([3, 2], dtype=int)
                tableholder.table(st.session_state['df'])

            jsn, tsv = convert_df(df)
            _, col2, col3, _ = st.columns([2, 1, 1, 2])
            col2.download_button(label="Download as JSON", data=jsn, file_name='results.json',mime='text/json')
            col3.download_button(label="Download as TSV", data=tsv, file_name='results.tsv', mime='text/csv')
        else:
            nresholder.markdown("#### No. of results: `0`")
            st.info(f'''
    No significant results 👀

                p3 = {st.session_state['pattern3']}''')

with st.sidebar:
    _, col2, _ = st.columns(3)
    clear = col2.button("Clear all 🗑️")
    st.subheader("")
    if clear:
        clear_data(['uploaded_files', 'pattern1', 'pattern2', 'pattern3', 'result'])
        st.experimental_rerun()
