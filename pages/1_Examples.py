import streamlit as st

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

st.markdown("## Examples :pencil2:")

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
st.markdown("**P1** : pattern{ V[Polarity=Neg, upos=VERB|AUX]}; e:X->V")
st.markdown('**P2** : pattern { V.form=re".*l" }')
st.markdown("**P3** : e.label; V.AnyFeat; X.AnyFeat")
st.markdown("---")
