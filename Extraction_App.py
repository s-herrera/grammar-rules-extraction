import grew
import extraction_tools as et
import streamlit as st
import pandas as pd
import tempfile
from typing import Tuple, Dict
from collections import namedtuple
import re

# -----------------------------------------

@st.experimental_memo(show_spinner=False)
def load_corpora(filenames: str) -> Tuple[Dict, int, int, int]:
    """
    Load corpus in a dictionary and by using Grew. Return corpora and its number of sentences and tokens.
    """
    # Variable 'filenames' is used to check if the upload files have changed
    with st.spinner('Loading treebank...'):

        with tempfile.NamedTemporaryFile(mode="wt", encoding="utf-8") as temp:
            for uploaded_file in st.session_state['upload_files']:
                temp.write(uploaded_file.getvalue().decode("utf-8"))

            treebank_idx = grew.corpus(temp.name)
            treebank = et.conllu_to_dict(temp.name)
            sentences, tokens = et.get_corpus_info(treebank)

    return treebank, treebank_idx, sentences, tokens

def convert_df(df):
    """
    Convert a dataframe into a tsv and a json.
    """
    jsn = df.to_json(orient="split", index=False, indent=4)
    tsv = df.to_csv(index=False, encoding = "utf-8", sep="\t")
    return jsn, tsv

def del_all_session_keys(excepts : list):
    for key in st.session_state.keys():
        if key not in excepts:
            del st.session_state[key]

# -----------------------------------------

# TODO
# Change MENU
# Information after P1&P2 in expander
# FIX title

st.set_page_config(
    page_title="Rules extraction", page_icon="📐", initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': '',
    #     'Report a bug': ",
    #     'About': "# This is a header. This is an *extremely* cool app!"},
    #layout="wide"
)

def _max_width_():
    # Change size of container. This could change every streamlit update
    max_width_str = f"max-width: 925px;"
    st.markdown(
        f"""
    <style>
    .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

_max_width_()

st.write(
    """
# 📐 Rules Extraction App
Extracting significant patterns from treebanks
"""
)

GrewPattern = namedtuple('GrewPattern', 'pattern without global_')

grew.init()

if "res" not in st.session_state.keys():
    st.session_state["res"] = []

# Useful links

with st.sidebar:
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

# Form 1
with st.form("form1"):

    uploaded_files = st.file_uploader("CoNLL/CoNLL-U accepted", type=[".conll", ".conllu"], accept_multiple_files=True, key="upload_files")
    filenames = [f.name for f in uploaded_files]

    submitted1 = st.form_submit_button("Upload")
    if submitted1:
        del_all_session_keys(["upload_files"])

    if not uploaded_files:
        st.info("Upload your files")
        st.stop()

    treebank, treebank_idx, sentences, tokens = load_corpora(filenames)

if uploaded_files:

    col1, col2, col3 = st.columns([1,1.4,5], gap="large")
    with col1:
        col1.metric("# Files", len(filenames))
    with col2:  
        col2.metric("# Sentences", sentences)
    with col3:
        col3.metric("# Tokens", tokens, help="Tokens such as 20-21 count as two")

    # Form2
    
    with st.form(key="form2"):

        st.subheader("Insert the first two querys")

        P1 = st.text_area("Pattern 1", value="", key="pattern1", help="pattern { e:X->Y; X[upos=NOUN]; Y[upos=ADJ] }", placeholder="pattern { e:X->Y; X[upos=NOUN]; Y[upos=ADJ] }", disabled=False)
        P2 = st.text_area("Pattern 2", value="", key="pattern2", help="pattern { Y << X }", placeholder="pattern { Y << X }", disabled=False)

        P1grew = et.build_GrewPattern(P1)
        P2grew = et.build_GrewPattern(P2)
        
        submitted2 = st.form_submit_button("Match")
        
        if submitted2:
            if P1 and P2:

                matchs, features = et.get_patterns_info(treebank_idx, treebank, P1grew)
                M, n = et.compute_fixed_totals(matchs, P1grew, P2grew, treebank_idx)
                st.session_state["M"] = M
                st.session_state["n"] = n
                st.session_state["matchs"] = matchs
                st.session_state["features"] = features

                if 'res' in st.session_state.keys():
                    st.session_state['res'] = []
            else:
                st.info("Complete both patterns!")

    if P1 and P2:

        M = st.session_state['M']
        n = st.session_state['n']

        col1, col2 = st.columns([1.3,2])
        with col1:
            st.caption("### Features of P1 nodes:")
            st.json(st.session_state['features'], expanded=False)
        with col2:
            df = pd.DataFrame({"pattern 2" : [n], " ¬ pattern 2" : [M-n], " total" : [M]}, index=["pattern 1"])
            st.table(df)

        # Form 3
        with st.form(key="form3"):
                        
            st.subheader("Insert the last query")
            P3 = st.text_area("Pattern 3 or Key(s) to cluster", value="", height=None, max_chars=None, key="pattern3", help=None, placeholder="pattern { X[upos=ADJ] }\n\tor\nX.upos; e.label", disabled=False)
            
            option = st.radio("Combination mode", ('Simple combinations', "All possible combinations"), horizontal=True)

            if option  == 'Simple combinations':
                combination_type = False
            else:
                combination_type = True
                
            submitted3 = st.form_submit_button("Get results")

            if submitted3:
                if P3:
                
                    # if combination_type:
                    #     st.warning("This could take a while...")

                    key_predictors = et.get_key_predictors(P1, P3)
                    st.session_state['keys'] = bool(key_predictors)

                    patterns = et.get_patterns(treebank, st.session_state["matchs"], P3, key_predictors, combination_type)
                    res = et.rules_extraction(treebank_idx, patterns, P1grew, P2grew, st.session_state["M"], st.session_state["n"])
                    st.session_state['res'] = res
                else:
                    st.info("Complete the pattern")


        if st.session_state['res']:

            if combination_type: 
                checkbox = st.checkbox("Get only the most significant subsets", value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
                if checkbox:
                    subsets = et.get_significant_subsets(st.session_state['res'])
                    res = [r for sset in subsets for r in st.session_state['res'] if sset in r]
                    res_df = pd.DataFrame(res, columns=["Pattern", "Significance", "Probability ratio", "# P2&P3", "% of P1&P2", "% of P1&P3"])
                    st.dataframe(res_df, width=925)
                else:
                    res_df = pd.DataFrame(st.session_state['res'], columns=["Pattern", "Significance", "Probability ratio", "# P2&P3", "% of P1&P2", "% of P1&P3"])
                    st.dataframe(res_df, width=925)
            else:
                res_df = pd.DataFrame(st.session_state['res'], columns=["Pattern", "Significance", "Probability ratio", "# P2&P3", "% of P1&P2", "% of P1&P3"])
                #res_df = pd.DataFrame(st.session_state['res'])
                #res_df.style.set_properties(subset=['Pattern'], **{'width': '600px'})
                st.dataframe(res_df, width=925)


            # with col3:
            #     st.download_button(
            #         label="Download data as TXT",
            #         data=tsv,
            #         file_name='results.txt',
            #         mime='text',
            #     )


            res_lst = res_df['Pattern'].to_list()
            res_lst.insert(0, '')

            option = st.selectbox(
                'Create a Grew-match link for the chosen pattern:',
                res_lst, index=0)
            if option:
                corpus = filenames[0].split(".")[0]

                # because keys patterns doesn't have the string "patterns {}". It's necessary to make the difference
                if st.session_state['keys']:
                    P3grew = et.build_GrewPattern(f"pattern {{ {option} }}")
                else:
                    P3grew = et.build_GrewPattern(option)
                
                link = et.get_Grewmatch_link(corpus, P1grew, P2grew, P3grew)
                st.info("The upload file name is used to query the corpus in Grew-match. It's possible that the selected corpus is not the correct one.")
                st.markdown(f'🔗[Grew-match link]({link})')

            jsn, tsv = convert_df(res_df)
            
            st.download_button(
                    label="Download as JSON",
                    data=jsn,
                    file_name='results.json',
                    mime='text/json',
                )

            st.download_button(
                    label="Download as TSV",
                    data=tsv,
                    file_name='results.tsv',
                    mime='text/csv',
                )

                # P3grew = et.build_GrewPattern(f"pattern {{ {option} }}")
                # P2whether = re.sub(r"without|pattern|global|{|}", "", et.grewPattern_to_string(P2grew))
                
                # enc_corpus = urllib.parse.quote(filenames[0].split(".")[0].encode('utf8'))
                # enc_pattern = urllib.parse.quote(et.grewPattern_to_string(P1grew, P3grew).encode('utf8'))
                # enc_whether = urllib.parse.quote(P2whether.strip().encode('utf-8'))
                
                # st.write(f"http://universal.grew.fr/?corpus={enc_corpus}&pattern={enc_pattern}&whether={enc_whether}")
                # print(enc_pattern)
                # print(enc_whether)

        else:
            res_df = pd.DataFrame([], columns=["Pattern", "Significance", "Probability ratio", "# P2&P3", "% of P1&P2", "% of P1&P3"])
            #res_df.style.set_properties(subset=['Pattern'], **{'width': '600px'})
            #res_df.style.set_properties({'width': '300px'})
            st.dataframe(res_df)

