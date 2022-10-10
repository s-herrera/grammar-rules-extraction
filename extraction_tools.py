import streamlit as st

import re
import os
import urllib
import time
import json
from collections import defaultdict, Counter, namedtuple
from itertools import combinations, chain
from typing import Dict, Iterable, List, Tuple, Set

import numpy as np
import grew
from scipy.stats import fisher_exact, hmean

# -------------- Fonctions --------------


GrewPattern = namedtuple('GrewPattern', 'pattern without global_')


class PatternError(Exception):
    pass


def conllu_to_dict(path: str) -> Dict:
    """
    Create a treebank (graph by dictionaries) form a conll/conllu-u file
    """
    with open(path) as f:
        conll = f.read().strip()

    trees = {}
    features = set()

    sentences = [x.split("\n") for x in conll.split("\n\n")]
    for sent in sentences:
        for line in sent:
            if line.startswith("#"):
                if re.match('# sent_id', line):
                    sent_id = line.split("=")[1].strip()
                    trees[sent_id] = {'0': {"form": "None", "lemma": "None", "upos": "None", "xpos": "None", "head": "None", "deprel": "None"}}
            else:
                token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = line.strip().split("\t")
                if "-" not in token_id:
                    trees[sent_id][token_id] = {"form": form, "lemma": lemma, "upos": upos, 'xpos': xpos, "head": head, "deprel": deprel, "deps": deps}
                    if feats != "_":
                        trees[sent_id][token_id].update(separate_column_values(feats))
                        features.update([x for x in separate_column_values(feats).keys()])
                    if misc != "_":
                        trees[sent_id][token_id].update(separate_column_values(misc))
    return trees, features


def separate_column_values(s: str) -> Dict:
    """
    It separates values from some columns of conll/conll-u files
    """
    values = [v.split("=") for v in s.split("|")]
    values_dict = {lst[0]: lst[1] for lst in values}
    return values_dict


def get_corpora_name(lst: list) -> str:

    with open('language_codes.json', encoding="utf-8") as f:
        code_to_lang = json.load(f)

    common_prefix = os.path.commonprefix(lst)
    splited = [x for x in re.split(r"-|_|\.", common_prefix) if x]
    if len(splited) <= 2:
        corpora = common_prefix
    else:
        lang_code, corpus, scheme, *_ = re.split(r"-|_|\.", common_prefix)
        corpora = f"{scheme.upper()}_{code_to_lang[lang_code]}-{corpus.upper()}"
    return corpora


def is_balanced(s: str) -> bool:
    """
    Check if the brackets ([,],{,}) are balanced.
    """
    brackets = [m for m in re.findall(r'\"[^\"]*\"|([\[\]\{\}])', s) if m]

    if len(brackets) % 2 != 0:
        return False

    pair_brackets = {'{': '}', '[': ']'}
    stack = []
    for char in brackets:
        if char in pair_brackets.keys():
            stack.append(char)
        else:
            if stack == []:
                return False
            open_bracket = stack.pop()
            if char != pair_brackets[open_bracket]:
                return False
    return stack == []


def is_valid_pattern(s: str) -> bool:
    """
    Check is the Grew commands (pattern, without, global) are well written.
    """
    s = re.sub("\n", "", s)  # erase newlines

    if not is_balanced(s):
        return PatternError("The curly or the square brackets are not balanced")

    grew_cmds = [x.split('}')[-1].strip() for x in s.split('{')]
    matchs = [re.match(r"^(pattern|without|global)$", w) for w in grew_cmds if w]
    if not all(matchs):
        return PatternError("The Grew commands (pattern, without, global) or the key patterns (e.g. X.upos) are not acceptable")


def build_GrewPattern(s: str) -> GrewPattern:
    """
    """
    s = re.sub("\n", "", s)  # erase newlines

    p = [x.strip() for x in re.findall(r"\bpattern\b\s*{(.+?)}", s) if x]
    w = [x.strip() for x in re.findall(r"\bwithout\b\s*{(.+?)}", s) if x]
    g = [x.strip() for x in re.findall(r"\bglobal\b\s*{(.+?)}", s) if x]
    pattern = GrewPattern(p, w, g)
    return pattern


def grewPattern_to_string(*patterns: GrewPattern) -> str:
    """
    """

    p = f"pattern {{{'; '.join(['; '.join(tpl.pattern) for tpl in patterns if tpl.pattern])}}}"
    w = " ".join([f"without {{{w}}}" for tpl in patterns for w in tpl.without if w])
    g = " ".join([f"global {{{g}}}" for tpl in patterns for g in tpl.global_ if g])
    pattern = p + w + g
    return pattern


def combine_simple_pattern(pat: str) -> str:
    """
    """
    p = [y.strip() for x in re.findall(r"pattern\s*{(.+?)}", pat) if x for y in x.split(";")]
    pwrset_p = [f"pattern {{ {'; '.join(x)} }}" for x in powerset(p) if x]
    w = [" ".join(x) for x in powerset(re.findall(r"without\s*{.+?}", pat))]
    g = [" ".join(x) for x in powerset(re.findall(r"global\s*{.+?}", pat))]

    res = set(" ".join(p) for r in range(3) for c in combinations([pwrset_p,w,g], r+1) for p in product(*c) if p)
    return res


def format_simple_pattern(*pattern: str) -> str:
    """
    It formats patterns into Grew valid patterns
    """
    res = ";".join(pattern)
    res = f"pattern {{ {res} }}"
    return res


def format_significance(p_value: float) -> int:
    """
    The negative exponent is taken as significance value when is possible.

    1.50e-50 = 50, 0.00050 = 3, 0.0 = inf
    """
    if "-" in str(p_value):
        significance = str(p_value).split("-")[1]
        significance = int(re.sub(r"^0", "", significance))
    elif p_value == 0:
        significance = float('inf')
    else:
        significance = re.search(r"\.(0+)", str(p_value)).group(1).count("0")
    return significance


def get_GrewMatch_link(filenames: str, p1: GrewPattern, p2: GrewPattern, p3: GrewPattern):

    corpora = get_corpora_name(filenames)
    p2whether = re.sub(r"without|pattern|global|{|}", "", grewPattern_to_string(p2))
    enc_corpus = urllib.parse.quote(corpora.encode('utf8'))
    enc_pattern = urllib.parse.quote(grewPattern_to_string(p1, p3).encode('utf8'))
    enc_whether = urllib.parse.quote(p2whether.strip().encode('utf-8'))
    link = f"http://universal.grew.fr/?corpus={enc_corpus}&pattern={enc_pattern}&whether={enc_whether}"
    return link


def powerset(iterable: Iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3). See Python itertools documentation
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def product(*args, repeat=1):
    """
    product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    See Python itertools documentation. Only "if pool" added.
    """
    pools = [tuple(pool) for pool in args if pool] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def get_corpus_info(treebank: Dict) -> Tuple[int, int]:
    """
    Get number of sentences and tokens.
    """
    sentences = len(treebank)
    tokens = len([token for v in treebank.values() for token in v.keys()]) # Count tokens 20-21 token as two
    return sentences, tokens


def get_patterns_info(treebank_idx: int, treebank: Dict, P1 : str) -> List[Dict] and  Dict[str, Set]:
    """
    Get P1 matchs and the nodes associated with each of its features.
    """
    matchs = grew.corpus_search(grewPattern_to_string(P1), treebank_idx)
    res = {n : set() for m in matchs for n in m["matching"]["nodes"].keys()}
    for m in matchs:
        for node, idx in m["matching"]["nodes"].items():
            res[node].update(k for k in treebank[m["sent_id"]][idx].keys() if k not in ("head", "deprel", "deps"))
    allfeatures = {k: sorted(v) for k, v in res.items()}
    return matchs, allfeatures


def compute_fixed_totals(matchs, P1, P2, treebank_idx):
    """
    Compute fixed totals of a contengcy table. M = P1 occurrences. n = P2 occurrences.
    """
    M = len([m for m in matchs])
    n = grew.corpus_count(pattern=grewPattern_to_string(P1, P2), corpus_index=treebank_idx)
    return M, n


def get_key_predictors(P1: str, P3: str, features: set) -> Dict[str, list]:
    """
    Get node : [key predictors] (key patterns) in a dictionary. The script doesn't accept mixed querys (with and without keys).
    """
    key_predictors = defaultdict(list)

    st.session_state['pattern_len'] = len(P3.split(';'))

    for pat in P3.split(';'):

        if re.search(r'^.+?\.\w+?$', pat):
            k, v = pat.strip().split(".")
            if v in ("label", "1", "2", "deep"):
                re_match = re.search(fr"{k}\s*:\s*(\w+?)\s*->\s*(\w+?)", P1)
                key_predictors[re_match.group(2)].append(["deprel", {"head" : re_match.group(1), "dep" : re_match.group(2)}, v])
            elif "AnyFeat" in v:
                node_pat = re.findall(fr"{k}\[.+?\]", P1)
                node_feats = re.findall(r"\w+(?==\w+)", " ".join(node_pat))
                features.difference_update([*node_feats, "Number[psor]", "Person[psor]", "Gender[psor]", "Clusivity[psor]", "Deixis[psor]"])
                key_predictors[k].extend([x for x in features])
            else:
                key_predictors[k].append(v)
    return dict(key_predictors)


def get_patterns(treebank: Dict, matchs: Dict, P3: str, key_predictors: Dict, option: bool) -> list or Dict[str, int]:
    """
    Build and count the potential patterns (str) to test them statistically according to P1, P2 and the chosen option.
    """
    if not key_predictors:
        if option:
            patterns = combine_simple_pattern(P3)
        else:
            patterns = [P3]
        return patterns

    counter = Counter()
    for m in matchs:
        lst = []
        for node, idx in m["matching"]["nodes"].items():
            if node in key_predictors:
                for pred in key_predictors[node]:
                    if isinstance(pred, list):
                        # it's a deprel with head and dep. Create pattern GOV-[deprel]->DEP
                        deprel = re.split(r"\:|@", treebank[m["sent_id"]][idx][pred[0]])
                        if pred[2] == "1":
                            pat = f'{pred[1]["head"]}-[1={deprel[0]}]->{node}'
                        elif pred[2] == "2" and ":" in treebank[m["sent_id"]][idx][pred[0]]:
                            pat = f'{pred[1]["head"]}-[2={deprel[1]}]->{node}'
                        elif pred[2] == "deep" and "@" in treebank[m["sent_id"]][idx][pred[0]]:
                            pat = f'{pred[1]["head"]}-[deep={deprel[-1]}]->{node}'
                        elif pred[2] == "label":
                            pat = f'{pred[1]["head"]}-[{treebank[m["sent_id"]][idx][pred[0]]}]->{node}'
                        else:
                            continue
                        lst.append(pat)
                    elif pred in treebank[m["sent_id"]][idx]:
                        pat = f'{node}[{pred}="{treebank[m["sent_id"]][idx][pred]}"]'
                        lst.append(pat)
        counter.update([x for x in powerset(lst)])
    if not option:
        patterns = {k: v for k, v in counter.items() if len(k) == st.session_state['pattern_len']}
        return patterns

    patterns = dict(counter)
    return patterns


def rules_extraction(treebank_idx: int, patterns: Dict, P1: GrewPattern, P2: GrewPattern, M: int, n: int) -> List[List] and Dict:

    # streamlit progress bar
    my_bar = st.progress(0.0)
    start = time.time()

    result = []
    tables = {}
    for i, pat in enumerate(patterns, start=1):

        my_bar.progress(i/len(patterns))

        # if it's a dict it has key pattern, on the contrary, it's a simple pattern
        if isinstance(patterns, dict):
            N = patterns[pat]
            pat = '; '.join(pat)
            P3 = build_GrewPattern(f"pattern {{ {pat} }}")
        else:
            P3 = build_GrewPattern(pat)
            N = grew.corpus_count(pattern=grewPattern_to_string(P1, P3), corpus_index=treebank_idx)
        k = grew.corpus_count(pattern=grewPattern_to_string(P1, P2, P3), corpus_index=treebank_idx)
        table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
        oddsratio = np.log(((table[0,0]+ 0.5)*(table[1,1]+0.5))/((table[0,1]+ 0.5)*(table[1,0]+0.5)))
        if oddsratio > 1:

            _, p_value = fisher_exact(table=table, alternative='greater')
            
        #Others measures
        #oddsratio = np.log(((table[0,0]+ 0.5)*(table[1,1]+0.5))/((table[0,1]+ 0.5)*(table[1,0]+0.5)))
        #pmi = np.log2((k/M)/((n/M)*(N/M)))
        #zscore = (k - ((N*n)/M)) / np.sqrt(((N*n)/M)*(1-(((N*n)/M)/N)))
        #h_mean  = hmean([(k/n),(k/N)])
        #probability_ratio = (k/N)/((n-k)/(M-N))

            if p_value < 0.01:
                percent_M1M2 = (k/n)*100
                percent_M1M3 = (k/N)*100
                result.append([pat, p_value, oddsratio, percent_M1M2, percent_M1M3])
                tables[pat] = table

    end = time.time()
    st.write(f"Time: {round(end - start, 3)}")
    return result, tables


def get_significant_subsets(lst: list):
    """
    Filter the results to obtain the most significant sub-patterns
    """
    res_sorted = sorted(lst, key=lambda x: (-x[1], len(x[0])))

    dropped, subsets = set(), set()

    for ires in res_sorted:
        ipattern = tuple(x.strip() for x in ires[0].split(";"))
        ipvalue, iPR = ires[1:3]
        for jres in res_sorted:
            jpattern = tuple(x.strip() for x in jres[0].split(";"))
            jpvalue, jPR = jres[1:3]
            if set(ipattern).issubset(jpattern) and jpattern not in dropped:
                if ipvalue < jpvalue:
                    dropped.add(jpattern)
                    subsets.add(ipattern)
                elif ipvalue == jpvalue:
                    if iPR > jPR:
                        dropped.add(jpattern)
                    elif iPR == jPR:
                        subsets.update([ipattern, jpattern])
                    else:
                        if ipattern != jpattern:
                            subsets.add(ipattern)

    subsets.difference_update(dropped)
    return subsets


# def get_GrewMatch_corpora():
#     """
#     """
#     SUD = 'https://surfacesyntacticud.github.io/data/'
#     req = requests.get(SUD)
#     soup = BeautifulSoup(req.content, 'html.parser')
#     res = [""]
#     for tag in soup.find_all(href=re.compile("corpus")):
#         corpusSUD = re.search(r'\?corpus=(.+?)"', str(tag)).group(1)
#         corpusUD = re.sub('SUD', 'UD', corpusSUD)
#         res.extend([corpusSUD, corpusUD])
#     return res
