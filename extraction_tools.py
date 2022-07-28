import re, time
import streamlit as st
import numpy as np
import grew
from collections import defaultdict, Counter, namedtuple
from itertools import combinations, chain
from typing import Dict, Iterable, List, Tuple, Set
from scipy.stats import fisher_exact
import urllib

# -------------- Fonctions --------------

GrewPattern = namedtuple('GrewPattern', 'pattern without global_')

def conllu_to_dict(path : str) -> Dict:
    """
    Create a treebank (graph by dictionaries) form a conll/conllu-u file
    """
    with open (path) as f:
        conll = f.read().strip()
    trees = {}
    sentences = [x.split("\n") for x in conll.split("\n\n")]
    for sent in sentences:
        for line in sent:
            if line.startswith("#"):
                if "sent_id" in line:
                    sent_id = line.split("=")[1].strip()
                    trees[sent_id] = {'0' : {"form" : "None", "lemma" : "None", "upos" : "None", "head" : "None", "deprel" : "None"} }
            else:
                token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = line.split("\t")
                if "-" not in token_id:
                    trees[sent_id][token_id] = {"form" : form, "lemma" : lemma, "upos" : upos, "head" : head, "deprel" : deprel}
                    if xpos != "_":
                        trees[sent_id][token_id].update(separate_column_values(xpos))
                    if feats != "_":
                        trees[sent_id][token_id].update(separate_column_values(feats))
                    if deps != "_":
                        trees[sent_id][token_id].update(separate_column_values(deps))
                    if misc != "_":
                        trees[sent_id][token_id].update(separate_column_values(misc))
    return trees

def separate_column_values(s : str) -> Dict:
    """
    It separates values from some columns of conll/conll-u files
    """
    values = [v.split("=") for v in s.split("|")]
    values_dict = {lst[0]:lst[1] for lst in values}
    return values_dict

def build_GrewPattern(s : str) -> GrewPattern:
    """
    """
    p = [x.strip() for x in re.findall(r"pattern\s*{(.+?)}", s) if x]
    w = [x.strip() for x in re.findall(r"without\s*{(.+?)}", s) if x]
    g = [x.strip() for x in re.findall(r"global\s*{(.+?)}", s) if x]
    pattern = GrewPattern(p, w, g)
    return pattern

def grewPattern_to_string(*patterns : GrewPattern) -> str:
    """
    """
    p = f"pattern {{{'; '.join(['; '.join(tpl.pattern) for tpl in patterns if tpl.pattern])}}}"
    w = " ".join([f"without {{{w}}}" for tpl in patterns for w in tpl.without if w])
    g = " ".join([f"global {{{g}}}" for tpl in patterns for g in tpl.global_ if g])
    pattern = p + w + g
    return pattern

def combine_simple_pattern(pat : str) -> str:
    """
    It formats patterns into Grew valid patterns
    """
    p = [y.strip() for x in re.findall(r"pattern\s*{(.+?)}", pat) if x for y in x.split(";")]
    pwrset_p = [f"pattern {{ {'; '.join(x)} }}" for x in powerset(p) if x]
    w = [" ".join(x) for x in powerset(re.findall(r"without\s*{.+?}", pat))]
    g = [" ".join(x) for x in powerset(re.findall(r"global\s*{.+?}", pat))]

    res = set(" ".join(p) for r in range(3) for c in combinations([pwrset_p,w,g], r+1) for p in product(*c) if p)
    return res


def format_simple_pattern(*pattern : str) -> str:
    """
    It formats patterns into Grew valid patterns
    """
    res = ";".join(pattern)
    res = f"pattern {{ {res} }}"
    return res

def format_significance(p_value : float) -> int:
    """
    The negative exponent is taken as significance value when is possible.
    0.0 = Inf
    1.50e-50 = 50
    0.00050 = 3
    """
    if "-" in str(p_value):
        significance = str(p_value).split("-")[1]
        significance = int(re.sub(r"^0", "", significance))
    elif p_value == 0:
        significance = np.inf
    else:
        significance = re.search(r"\.(0+)", str(p_value)).group(1).count("0")
    return significance

def get_Grewmatch_link(corpus : str, P1 : GrewPattern, P2 : GrewPattern, P3 : GrewPattern):
    P2whether = re.sub(r"without|pattern|global|{|}", "", grewPattern_to_string(P2))
    enc_corpus = urllib.parse.quote(corpus.encode('utf8'))
    enc_pattern = urllib.parse.quote(grewPattern_to_string(P1, P3).encode('utf8'))
    enc_whether = urllib.parse.quote(P2whether.strip().encode('utf-8'))
    link = f"http://universal.grew.fr/?corpus={enc_corpus}&pattern={enc_pattern}&whether={enc_whether}"
    return link

def powerset(iterable : Iterable):
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

def get_patterns_info(treebank_idx: int, treebank: Dict, P1 : str) -> Tuple[List[Dict], Dict[str, Set]]:
    """
    Get P1 matchs and the nodes associated with each of its features.
    """
    matchs = grew.corpus_search(grewPattern_to_string(P1), treebank_idx)
    res = {n : set() for m in matchs for n in m["matching"]["nodes"].keys()}
    for m in matchs:
        for node, idx in m["matching"]["nodes"].items():
            res[node].update(k for k in treebank[m["sent_id"]][idx].keys() if k not in ("head", "deprel"))
    return matchs, {k : sorted(v) for k, v in res.items()}

def compute_fixed_totals(matchs, P1, P2, treebank_idx):
    """
    Compute fixed totals of a contengcy table. M = P1 occurrences. n = P2 occurrences.
    """
    M = len([m for m in matchs])
    n = grew.corpus_count(pattern = grewPattern_to_string(P1, P2), corpus_index = treebank_idx)
    return M, n

def get_key_predictors(P1: str, P3: str) -> Dict[str, list]:
    """
    Get node : [key predictors] (key patterns) in a dictionary. The script doesn't accept mixed querys (with and without keys).
    """
    key_predictors = defaultdict(list)
    for pat in P3.split(';'):
        if re.search(r'^.+?\.\w+?$', pat):
            k, v = pat.strip().split(".")
            if "label" in v:
                re_match = re.search(fr"{k}\s*:\s*(\w+?)->(\w+?)", P1)
                key_predictors[re_match.group(2)].append(["deprel", {"head" : re_match.group(1), "dep" : re_match.group(2)}])
            else:
                key_predictors[k].append(v)
    return dict(key_predictors)

def get_patterns(treebank : Dict, matchs : Dict , P3 : str, key_predictors : Dict, option : bool) -> list or Dict[str, int]:
    """
    Build and count the potential patterns (str) to test them statistically according to P1, P2 and the chosen option.
    """
    if not key_predictors:
        if option:
            patterns = combine_simple_pattern(P3)
        else:
            patterns = [P3]
        return patterns

    pattern_len = sum([len(v) for v in key_predictors.values()])

    res = []
    for m in matchs:
        lst = []
        for node, idx in m["matching"]["nodes"].items():
            if node in key_predictors:
                for pred in key_predictors[node]:
                    if isinstance(pred, list):
                        # it's a deprel with head and dep. Create pattern GOV-[deprel]->DEP
                        pat = f'{pred[1]["head"]}-[{treebank[m["sent_id"]][idx][pred[0]]}]->{node}'
                        lst.append(pat)
                    elif pred in treebank[m["sent_id"]][idx]:
                        # pattern Node[feature=value]
                        pat = f'{node}[{pred}="{treebank[m["sent_id"]][idx][pred]}"]'
                        lst.append(pat)
        if option:
            res.extend(["; ".join(v) for v in powerset(lst) if v])
        if not option and len(lst) == pattern_len:
            res.append("; ".join(lst))

    patterns = Counter(res)
    return dict(patterns)

def rules_extraction(treebank_idx : int, patterns : Dict, P1 : GrewPattern, P2: GrewPattern, M : int, n : int) -> List[List]:

    # streamlit progress bar
    my_bar = st.progress(0.0)
    start = time.time()

    res = []

    for i, pat in enumerate(patterns, start=1):

        my_bar.progress(i/len(patterns))

        # if it's a dict it has key pattern, on the contrary, it's a simple pattern
        if isinstance(patterns, dict):
            N = patterns[pat]
            P3 = build_GrewPattern(f"pattern {{ {pat} }}")
        else:
            P3 = build_GrewPattern(pat)
            N = grew.corpus_count(pattern = grewPattern_to_string(P1, P3), corpus_index = treebank_idx)
        
        # For the lemmas ?
        # if N <= 1:
        #     continue

        k = grew.corpus_count(pattern = grewPattern_to_string(P1, P2, P3), corpus_index = treebank_idx)
        table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
        _, p_value = fisher_exact(table = table, alternative='greater')
        if p_value < 0.01:
            significance = format_significance(p_value)
            percent_M1M2 = round((k/n)*100, 3)
            percent_M1M3 = round((k/N)*100, 3)
            probability_ratio = round((k/N)/((n-k)/(M-N)), 3)
            res.append([pat, significance, probability_ratio, k, percent_M1M2, percent_M1M3])

    end = time.time()
    st.write(f"Time: {round(end - start, 3)}")
    return res


def get_significant_subsets(res : list):

    res_sorted_len = sorted(res, key = lambda x: len(x[0]))
    # On filtre les résultats pour garder les motifs et sous-motifs plus significatifs

    subsets, visited = set(), set()

    for res in res_sorted_len:
        to_keep = set()
        pat, pvalue, PR, _, _, _ = res
        for xres in res_sorted_len:
            if set(pat).issubset(set(xres[0])):
                xpat, xpvalue, xPR, _, _, _ = xres
                if xpvalue < pvalue:
                    to_keep.add(xpat)
                elif xpvalue == pvalue and xPR > PR:
                    to_keep.add(xpat)
                elif xpvalue == pvalue and xPR == PR:
                    if pat != xpat:
                        visited.add(xpat)
                    else:
                        to_keep.add(pat)
                else:
                    visited.add(xpat)
        subsets.update(to_keep)
    subsets.difference_update(visited)
    return subsets
