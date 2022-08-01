import grew
import extraction_tools as et
import streamlit as st
import pandas as pd
import tempfile
from typing import Tuple, Dict
from collections import namedtuple
# -----------------------------------------

@st.experimental_memo(show_spinner=False, persist='disk')
def load_corpora(files: dict) -> Tuple[Dict, int, int, int]:
    """
    Load corpus in a dictionary and by using Grew. Return corpora and its number of sentences and tokens.
    """
    # Variable 'files' is used to check if the upload files have changed
    with st.spinner('Loading treebank...'):
        with tempfile.NamedTemporaryFile(mode="wt", encoding="utf-8") as temp:
            for uploaded_file in st.session_state['upload_files']:
                f = uploaded_file.getvalue().decode("utf-8")
                for line in f:
                    temp.write(line)
            temp.seek(0)
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

def get_dataframe(lst : list) -> pd.DataFrame:
    df = pd.DataFrame(lst, columns=["Pattern", "Significance", "Probability ratio", "# P2&P3", "% of P1&P2", "% of P1&P3"])
    df['Significance'] = df['Significance'].apply(lambda x: et.format_significance(x))
    df = df.sort_values('Significance', ascending=False)
    df = df.reset_index(drop=True)
    style_df = df.style.format({"Significance" : "{:.0f}", "% of P1&P2": "{:.4}", "% of P1&P3": "{:.4}", "Probability ratio" : "{:.4}"})
    style_df = style_df.set_table_styles([dict(selector="th", props=[('max-width', '300px'),
                                ('text-overflow', 'ellipsis'), ('overflow', 'hidden')])])

    return style_df

def del_all_session_keys(excepts : list):
    for key in st.session_state.keys():
        if key not in excepts:
            del st.session_state[key]

# -----------------------------------------

# TODO
# Change MENU
# Information after P1&P2 in expander
# FIX title

def _max_width_():
    # Change size of container. This could change every streamlit update
    max_width_str = f"max-width: 1100px;"
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


st.set_page_config(
    page_title="Rules extraction", page_icon="📐", initial_sidebar_state="expanded",
)

_max_width_()

hide_menu_style = """
        <style>
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            ul[data-testid=main-menu-list] > ul:nth-of-type(4) > li:nth-of-type(1) {display: none;}
            ul[data-testid=main-menu-list] > div:nth-of-type(2) {display: none;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

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

corpora = et.get_GrewMatch_corpora()

# pd.options.display.max_colwidth = 800
# pd.options.display.expand_frame_repr = True
# pd.set_option("colheader_justify", "right")

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

# Form 1

with st.form("form1"):

    uploaded_files = st.file_uploader("CoNLL/CoNLL-U accepted", type=[".conll", ".conllu"], accept_multiple_files=True, key="upload_files")
    files = {f.name : {'size' : f.size} for f in uploaded_files}

    submitted1 = st.form_submit_button("Upload")

    if submitted1:
        del_all_session_keys(excepts = ["upload_files"])
        
    if not uploaded_files:
        st.info("Upload your files")
        load_corpora.clear()
        st.stop()

    treebank, treebank_idx, sentences, tokens = load_corpora(files)

if st.session_state['upload_files']:

    col1, col2, col3 = st.columns([1,1.4,5], gap="large")
    with col1:
        col1.metric("# Files", len(files))
    with col2:  
        col2.metric("# Sentences", sentences)
    with col3:
        col3.metric("# Tokens", tokens, help="It includes multiword tokens (e.g. n-n+1 indexed tokens)")

    # Form2
    
    with st.form(key="form2"):

        st.subheader("Insert the first two querys")

        p1_help = '''
It accepts regular patterns.

`pattern {e:X->Y; X[upos=NOUN]; Y[upos=ADJ]}`

`pattern {X-[mod@relcl]->V} without {V->Y; Y[form=qui]}`
'''

        p2_help = '''
It accepts regular patterns.

`pattern {Y << X}`

`pattern {e:H->X}`
'''

        P1 = st.text_area("Pattern 1", value="", key="pattern1", help=p1_help)
        P2 = st.text_area("Pattern 2", value="", key="pattern2", help=p2_help)

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
                
            else:
                st.info("Complete both patterns!")

    if P1 and P2:

        M = st.session_state['M']
        n = st.session_state['n']

        col1, col2 = st.columns([1.3,2])
        with col1:
            st.caption("### Features of P1's nodes:")
            st.json(st.session_state['features'], expanded=False)
        with col2:
            df = pd.DataFrame({"pattern 2" : [n], " ¬ pattern 2" : [M-n], " total" : [M]}, index=["pattern 1"])
            st.table(df)

        # Form 3
        with st.form(key="form3"):

            p3_help="""
It's possible to use simple patterns or series of keys.

**Simple pattern:** `pattern {Y[NumType=Ord]}`

**Keys:** `e.label; X.lemma`

**Special key:** `X.AnyFeat`

The AnyFeat key includes any feature of the choosen node except those selected in the P1 and P2 and those listed below:  
**lemma, form, CorrectForm, wordform, SpaceAfter, xpos, Person[psor], Number[psor]**
 """


            option_help= """
**All possible combinations**

For the **pattern {X[upos=NOUN]; X-[subj]->Y}**, it will look for:

`X[upos=NOUN]`, `X-[subj]->Y` and `X[upos=NOUN]; X-[subj]->Y` patterns

For the **pattern {X[upos=ADJ]} without {X[Gender]}**, it will look for:

`X[upos=ADJ]`, `without {X[Gender]}` and `X[upos=ADJ] without {X[Gender]}` patterns

It works also for the AnyFeat key.
"""          
            
            st.subheader("Insert the last query")
            P3 = st.text_area("Pattern 3 or Key(s) to cluster", value="", height=None, max_chars=None, key="pattern3", help=p3_help)
            
            option = st.radio("Combination mode", ('Simple combinations', "All possible combinations"), horizontal=True, help=option_help)

            if option  == 'Simple combinations':
                combination_type = False
            else:
                combination_type = True
                
            submitted3 = st.form_submit_button("Get results")

            if submitted3:
                if P3:
                    key_predictors = et.get_key_predictors(P1, P3, st.session_state['features'])
                    st.session_state['keys'] = bool(key_predictors)
                    patterns = et.get_patterns(treebank, st.session_state["matchs"], P3, key_predictors, combination_type)
                    res = et.rules_extraction(treebank_idx, patterns, P1grew, P2grew, st.session_state["M"], st.session_state["n"])
                    st.session_state['res'] = res

                else:
                    st.info("Complete the last pattern!")


        if st.session_state['res']:
            st.subheader("")    
            if combination_type: 

                checkbox_help="""
                Filter to get the most significant subsets of patterns without losing potentially important information.
                
                1. It keeps the most significant subsets.

                2. If two patterns are equally significant, it compares the probability ratio (PR) keeping the pattern with the higher PR

                3. If the PR is also equal, it keeps both patterns.

                """

                checkbox = st.checkbox("Get only the most significant subsets", help=checkbox_help)
                if checkbox:
                    subsets = et.get_significant_subsets(st.session_state['res'])
                    res = [r for sset in subsets for r in st.session_state['res'] if sset == tuple(x.strip() for x in r[0].split(";"))]
                    style_df = get_dataframe(res)
                    st.dataframe(style_df)
                else:
                    style_df = get_dataframe(st.session_state['res'])
                    st.dataframe(style_df)
            else:
                style_df = get_dataframe(st.session_state['res'])
                st.dataframe(style_df)

            res_lst = style_df.data['Pattern'].to_list()
            res_lst.insert(0, '')
            st.markdown("***")
            col1, col2, col3, col4 = st.columns([1.3,1.1,1.1,1])
            with col1:
                st.markdown("")
                st.markdown("")
                st.subheader("Grew-match link ➡️")
            with col2:
                corpus = st.selectbox('Select the corpus :', corpora)
            with col3:
                choice = st.selectbox('Choose the pattern:', res_lst, index=0)
            with col4:

                if corpus and choice:
                    # because keys patterns doesn't have the string "patterns {}". It's necessary to make the difference
                    if st.session_state['keys']:
                        P3grew = et.build_GrewPattern(f"pattern {{ {choice} }}")
                    else:
                        P3grew = et.build_GrewPattern(choice)
                
                    link = et.get_Grewmatch_link(corpus, P1grew, P2grew, P3grew)
                    st.markdown("")
                    st.markdown(f'## 🔗 [link]({link})')
                else:
                    st.markdown("")
                    st.markdown("")
                    st.markdown("")
            st.markdown("***")
            jsn, tsv = convert_df(style_df.data)
            col1, col2, col3, col4 = st.columns([1.1,0.3,0.8,1])
            with col1:
                st.markdown("")
                st.markdown("")
                st.subheader("Download results ⬇️")
            with col3:
                st.markdown("")
                st.subheader("")
                st.download_button(
                        label="Download as JSON",
                        data=jsn,
                        file_name='results.json',
                        mime='text/json',
                    )
            with col4:
                st.markdown("")
                st.subheader("")
                st.download_button(
                        label="Download as TSV",
                        data=tsv,
                        file_name='results.tsv',
                        mime='text/csv',
                    )
            st.markdown("***")

        else:
            st.empty()
