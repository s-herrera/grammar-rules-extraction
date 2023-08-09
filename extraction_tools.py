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
from grewpy import Request
from grewpy.grs import RequestItem
from scipy.stats import fisher_exact, hmean

# -------------- Fonctions --------------


GrewPattern = namedtuple('GrewPattern', ['pattern', 'without', 'with_', 'global_'])

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
                        trees[sent_id][token_id].update(separate_column_values(feats, sent_id))
                        features.update([x for x in separate_column_values(feats, sent_id).keys()])
                    if misc != "_":
                        trees[sent_id][token_id].update(separate_column_values(misc, sent_id))
    return trees, features


def separate_column_values(s: str, sent_id: str) -> Dict:
    """
    It separates values from some columns of conll/conll-u files
    """
    values_dict = {}
    if "=" in s:
        try:
            values = [v.split("=") for v in s.split("|")]
            values = [[re.sub(r"\[(.+?)\]", r"__\1", i[0]), i[1]] for i in values]
            values_dict = {lst[0]: lst[1] for lst in values}
        except Exception:
            st.warning(f"The file contains an error in sentence {sent_id} that concerns one of the values of the string '{s}'. These values are ignored.")
    return values_dict


def get_corpora_name(lst: list) -> str:

    with open('language_codes.json', encoding="utf-8") as f:
        code_to_lang = json.load(f)

    common_prefix = os.path.commonprefix(lst)
    splited = [x for x in re.split(r"-|_|\.", common_prefix) if x]
    if len(splited) <= 2:
        corpora = common_prefix
    else:
        lang_code, corpus_name, scheme, *_ = re.split(r"-|_|\.", common_prefix)
        try:
            corpora = f"{scheme.upper()}_{code_to_lang[lang_code]}-{corpus_name.upper()}"
        except KeyError:
            corpora = f"{scheme.upper()}_{lang_code}-{corpus_name.upper()}"
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
    matchs = [re.match(r"^(pattern|without|with|global)$", w) for w in grew_cmds if w]
    if not all(matchs):
        return PatternError("The Grew commands (pattern, without, with, global) or the key patterns (e.g. X.upos) are not acceptable")


# def build_GrewPattern(s: str) -> GrewPattern:
#     """
#     """
#     s = re.sub("\n", "", s)  # erase newlines

#     pattern = [x.strip() for x in re.findall(r"\bpattern\b\s*{(.+?)}", s) if x]
#     without = [x.strip() for x in re.findall(r"\bwithout\b\s*{(.+?)}", s) if x]
#     with_ = [x.strip() for x in re.findall(r"\bwith\b\s*{(.+?)}", s) if x]
#     global_ = [x.strip() for x in re.findall(r"\bglobal\b\s*{(.+?)}", s) if x]
#     request = GrewPattern(pattern, without, with_, global_)
#     return request


def build_request(s: str) -> Request:
    """
    """
    s = s.strip()  # erase newlines

    pattern = ";".join([x.strip() for x in re.findall(r"\bpattern\b\s*{(.+?)}", s) if x])
    request = Request(pattern)

    without = ";".join([x.strip() for x in re.findall(r"\bwithout\b\s*{(.+?)}", s) if x])
    if without:
        request.without(without)

    with_ = ";".join([x.strip() for x in re.findall(r"\bwith\b\s*{(.+?)}", s) if x])
    if with_:
        request.with_(with_)

    # global_ = ";".join([x.strip() for x in re.findall(r"\bglobal\b\s*{(.+?)}", s) if x])
    # if global_:
    #     request.global_(global_)

    return request


def grewPattern_to_string(*patterns: GrewPattern) -> str:
    """
    """
    pattern = " ".join([f"pattern {{{w}}}" for tpl in patterns for w in tpl.pattern if w])
    without = " ".join([f"without {{{w}}}" for tpl in patterns for w in tpl.without if w])
    with_ = " ".join([f"with {{{w}}}" for tpl in patterns for w in tpl.with_ if w])
    global_ = " ".join([f"global {{{g}}}" for tpl in patterns for g in tpl.global_ if g])
    request = pattern + without + with_ + global_
    return request


def combine_simple_pattern(pat: str) -> str:
    """
    """
    pattern = [y.strip() for x in re.findall(r"pattern\s*{(.+?)}", pat) if x for y in x.split(";")]
    pwrset_p = [f"pattern {{ {'; '.join(x)} }}" for x in powerset(pattern) if x]
    without = [" ".join(x) for x in powerset(re.findall(r"without\s*{.+?}", pat))]
    with_ = [" ".join(x) for x in powerset(re.findall(r"with\s*{.+?}", pat))]
    global_ = [" ".join(x) for x in powerset(re.findall(r"global\s*{.+?}", pat))]

    request = set(" ".join(p) for r in range(3) for c in combinations([pwrset_p, with_, without, global_], r+1) for p in product(*c) if p)
    return request


def format_simple_pattern(*pattern: str) -> str:
    """
    It formats patterns into Grew valid patterns
    """
    req = ";".join(pattern)
    req = f"pattern {{ {req} }}"
    return req


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


def get_GrewMatch_link(filenames: str, p1: str, p2: str, p3: str):

    # corpora_name = get_corpora_name(filenames)
    corpora_name = filenames[0]
    p2whether = re.sub(r"pattern|without|with|global|{|}", "", p2)
    enc_corpus = urllib.parse.quote(corpora_name.encode('utf8'))
    enc_pattern = urllib.parse.quote(str(p1 + " " + p3).encode('utf8'))
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


def get_patterns_info(corpus, treebank: Dict, P1 : str) -> List[Dict] and  Dict[str, Set]:
    """
    Get P1 matchs and the nodes associated with each of its features.
    """
    request = build_request(P1)
    matchs = corpus.search(request)
    res = {n : set() for m in matchs for n in m["matching"]["nodes"].keys()}
    for m in matchs:
        for node, idx in m["matching"]["nodes"].items():
            res[node].update(k for k in treebank[m["sent_id"]][idx].keys() if k not in ("head", "deprel", "deps"))
    allfeatures = {k: sorted(v) for k, v in res.items()}
    return matchs, allfeatures


def compute_fixed_totals(matchs, P1, P2, corpus):
    """
    Compute fixed totals of a contengcy table. M = P1 occurrences. n = P2 occurrences.
    """
    M = len([m for m in matchs])
    req1 = build_request(P1)
    req2 = build_request(P2)
    n = corpus.count(Request(req1, req2))

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
                features.difference_update([*node_feats])
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


def rules_extraction(corpus, patterns: Dict, P1: str, P2: str, M: int, n: int) -> List[List] and Dict:

    # streamlit progress bar
    my_bar = st.progress(0.0)
    start = time.time()

    result = []
    tables = {}

    req1 = build_request(P1)
    req2 = build_request(P2)


    for i, pat in enumerate(patterns, start=1):

        my_bar.progress(i/len(patterns))
        # if it's a dict it has key pattern, on the contrary, it's a simple pattern
        if isinstance(patterns, dict):
            N = patterns[pat]
            pat = '; '.join(pat)

            req3 = build_request(f"pattern {{ {pat} }}")
            # P3 = build_GrewPattern(f"pattern {{ {pat} }}")
        else:
            req3 = build_request(pat)
            # P3 = build_GrewPattern(pat)

# corpus.count(Request("; ".join(P1.pattern + P2.pattern)))

            N = corpus.count((Request(req1, req2)))
            # N = grew.corpus_count(pattern=grewPattern_to_string(P1, P3), corpus_index=treebank_idx)

        k = corpus.count((Request(req1, req2, req3)))
        # k = grew.corpus_count(pattern=grewPattern_to_string(P1, P2, P3), corpus_index=treebank_idx)
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
                p_value = -np.log10(p_value)
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
