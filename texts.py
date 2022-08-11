ABOUT = '''
**Extracting significant patterns from treebanks**


It allows to query a treebank to statistically compare the distribution of a
chosen pattern against others. We use Grew-match query language
to interrogate the treebanks.

On the basis of three query patterns, we search for the conditions (pattern P3)
that trigger a linguistic phenomenon (pattern P2),
in the initial search space (pattern P1). We performed a significance test
to evaluate the distribution obtained from these three patterns.

If it is your first time, go through the **tutorial**.

The **right top menu** allows you to modify how the site looks.
'''

P1P2_HELP = '''
They accept normal Grew patterns.

noun-adjective inversion:

p1 : `pattern {e:X->Y; X[upos=NOUN]; Y[upos=ADJ]}`

p2 : `pattern {Y << X}`

noun number agreement:

p1 : `pattern {e:X->Y; X[upos=NOUN]}`

p2 : `pattern {X.Number = Y.Number}`

'''

P3_HELP = '''
It is possible to use simple patterns or series of keys.

**Simple pattern:** `pattern {Y[NumType=Ord]}`

**Keys:** `e.label; X.lemma`

**Special key:** `X.AnyFeat`

The AnyFeat key includes any feature of the chosen node except those selected in the P1 and those listed below:
**lemma, form, CorrectForm, wordform, SpaceAfter, xpos, Person[psor], Number[psor]**
'''


OPTION_HELP = '''
**All possible combinations**

For a normal pattern, it will look for all possibles subpatterns.

1) **pattern {X[upos=NOUN]; X-[subj]->Y}**:

`X[upos=NOUN]`, `X-[subj]->Y` and `X[upos=NOUN]; X-[subj]->Y`

2) **pattern {X[upos=ADJ]} without {X[Gender]}**

`X[upos=ADJ]`, `without {X[Gender]}` and `X[upos=ADJ] without {X[Gender]}`

For key series, it combines all possible queried values a node has, if they can be combined.

1) Assuming that X is an adjective from French, for the keys **X.Number; X.Gender**, it will look for:

`X[Number=Sing]`, `X[Number=Plur]`, `X[Gender=Masc]`, `X[Gender=Fem]`,

`X[Number=Sing]; X[Gender=Masc]`, `X[Number=Sing]; X[Gender=Fem]`, etc.

It works also for the AnyFeat key.
'''

CHECKBOX_HELP = '''
Filter to get the most significant subsets of patterns without losing potentially important information.

1. It keeps the most significant subsets.

2. If two patterns are equally significant, it compares the probability ratio (PR) keeping the pattern with the higher PR

3. If the PR is also equal, it keeps both patterns.
'''
