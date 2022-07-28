# Grammar rules extraction App

This is an application to extract significant patterns, potential grammar rules, from treebanks using statistical methods.

## Description

It allows to query a treebank to statistically compare the distribution of the chosen patterns against others. 
We use [Grew-match](http://match.grew.fr/) query language to interrogate the treebanks. 
The application is build using the `streamlit` library. 


This work was done as part of an internship in the [ANR Autogramm project](https://autogramm.github.io/). It is also the implementation 
of my master's thesis (pluriTAL master's program).

## Getting Started

### Local use

1. Download this repository either by downloading and unzipping it from the website or by cloning it with git.

```bash
git clone https://github.com/santiagohy/grammar-rules-extraction.git
cd grammar-rules-extraction
```

2. Follow the [instructions up to the _Step 3_](https://grew.fr/usage/install/) to install grew and grewpy after running `apt-get update && upgrade`

3. Create a virtual environment in which to run the app.

```bash
python3 -m virtualenv .venv
source .venv/bin/activate
```

4. Install the dependencies.

```bash
pip3 install -U -r requirements.txt
```
This program was developed in a python 3.8 environment


5. Run the app.

```python
python3 -m streamlit run Extraction_App.py
```

## Contributing
For any problem or suggestion, posting issues is welcome.

## Useful links 🔗

To get the most out of it, we recommend to complement it with the following resources:

[**Grew-match**](http://match.grew.fr/)  

[**Universal tables**](http://tables.grew.fr/)

[**UD Guidelines**](https://universaldependencies.org/guidelines.html)

[**SUD Guidelines**](https://surfacesyntacticud.github.io/guidelines/u/)

[**UD Corpora**](https://universaldependencies.org/#download)

[**SUD Corpora**](https://surfacesyntacticud.github.io/data/)


## Acknowledgments

Thanks to Sylvain and Bruno
