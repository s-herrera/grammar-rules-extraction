import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Rules extraction", page_icon="📐",
    initial_sidebar_state="expanded",
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

st.header("Tutorial and more 📝")

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

tab1, tab2, tab3 = st.tabs(['Tutorial 📌', 'Examples ✏️', 'About ℹ️'])

with tab1:
    st.markdown("""

Languages have internal restrictions that are called grammar rules.
The set of rules makes a grammar. The study of a language requires a deep analysis of its grammar
but describing its rules is a time-consuming task.

We propose a system of exploration and extraction of significant patterns, potential grammar rules, using treebanks. Two hypotheses are assumed:

A grammar rule is :
- An over-represented pattern compared to others
- A set of conditions that trigger a particular pattern in a statistically significant way

We use [Grew-match](http://match.grew.fr/) system to query a treebank through patterns and the Fisher's Exact Test to evaluate the statistical significance of a pattern, testing their independance or not.

Formalization of a grammar rule:
""")
    st.markdown("---")
    _, col2, col3, _ = st.columns([1,2,2,1])
    col2.markdown("##### X ⇒ Y | C")
    col2.markdown("**C** are the conditions that for a given **X** trigger the linguistic phenomenon **Y**")
    col3.markdown("##### P1 ⇒ P2 | P3")
    col3.markdown("""
    **Pattern 1**: start search area   
    **Pattern 2**: dependent variable   
    **Pattern 3**: explanatory variable
    """)

    st.markdown("---")
    st.markdown("We want to reject the null hypothesis if the probability of obtaining the observed distribution of the patterns, or even one more extreme, in a given treebank, is lower than the critical value (p-value < 0.01)")
    st.markdown("#### Search area")
    st.image(r"static/images/search-space.jpg")

    st.markdown("#### Metrics")
    st.markdown(r"""
Metrics to rank the patterns:

**Significance**: We use the negative exponent of the p-value. If the p-value is equal to 0, we use "Infinity".

**Probability ratio**: $$PR = \frac{\#P1\&P2/\#P1\&P3}{\#P1\&P2/\#P1\&¬P3}$$
""")
with tab2:
    st.markdown("#### nominal subject-verb inversion ")
    st.markdown("**P1** : pattern {e:H->X; X-[subj]->Y; Y[upos=NOUN|PROPN]}")
    st.markdown("**P2** : pattern { X << Y}")
    st.markdown("**P3** : e.label")
    st.markdown("---")

    st.markdown("#### noun-adjective inversion")
    st.markdown("**P1** : pattern {e:X->Y; X[upos=NOUN]; Y[upos=ADJ]}")
    st.markdown("**P2** : pattern { Y << X}")
    st.markdown("**P3** : Y.AnyFeat")
    st.markdown("---")

    st.markdown("#### noun number agreement")
    st.markdown("**P1** : pattern {e:X->Y; X[upos=NOUN]; X[Number=Sing|Plur]; Y[Number=Sing|Plur]}")
    st.markdown("**P2** : pattern { X.Number = Y.Number}")
    st.markdown("**P3** : e.label; X.AnyFeat; Y.AnyFeat")
    st.markdown("---")

    st.markdown("#### ul negation of verbs (wolof)")
    st.markdown("**P1** : pattern{ V[Polarity=Neg, upos=VERB|AUX]; e:X->V}")
    st.markdown('**P2** : pattern { V.form=re".*l" }')
    st.markdown("**P3** : e.label; V.AnyFeat; X.AnyFeat")

with tab3:
    col1, col2 = st.columns([2, 1])
    col1.subheader("")
    col1.markdown("This work was done as part of an internship in the [ANR Autogramm project](https://autogramm.github.io/) (Induction of descriptive grammars from annotated corpora).")
    col2.image("https://autogramm.github.io/images/logo_autogramm.jpg", use_column_width=True,)

