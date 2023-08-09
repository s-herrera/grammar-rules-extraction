# Grammar rules extraction App

This is an application to explore and extract significant patterns, potential grammar rules, from treebanks using statistical methods.

## Description

It allows to query a treebank to statistically compare the distribution of a chosen pattern against others. 
We use [Grew-match](http://match.grew.fr/) query language to interrogate the treebanks. 
The application is build using the `streamlit` library. 

On the basis of three query patterns, we search for the conditions (pattern P3) that trigger the linguistic phenomenon expressed by the pattern P2, in the initial search space (pattern P1). We performed a significance test to evaluate the distribution obtained from these three patterns. 

This work was done as part of an internship in the [ANR Autogramm project](https://autogramm.github.io/). It is also the implementation 
of my master's thesis (pluriTAL master's program).

## Getting Started

### Local use

1. Download this repository either by downloading and unzipping it from the website or by cloning it with git.

```bash
git clone https://github.com/santiagohy/grammar-rules-extraction.git
cd grammar-rules-extraction
```

2. Follow this [instructions up to the _Step 3_](https://grew.fr/usage/install/) to install **grew** and **grewpy** after running `sudo apt update && sudo apt upgrade` or just run the next lines.

    - **Linux**

      ```bash
      sudo apt update && sudo apt upgrade
      sudo apt install opam
      sudo apt install wget m4 unzip librsvg2-bin curl bubblewrap
      opam init
      opam switch create 4.13.1 4.13.1
      eval $(opam env --switch=4.13.1)
      opam remote add grew "http://opam.grew.fr"
      opam install grew grewpy_backend
      ```

    - **Mac OS X**

      - Install [XCode](https://developer.apple.com/xcode/)

      - Install [Brew](https://brew.sh/)

      ```
      brew install aspcud
      brew install opam
      opam init
      opam switch create 4.13.1 4.13.1
      eval $(opam env --switch=4.13.1)
      opam remote add grew "http://opam.grew.fr"
      opam install grew grewpy_backend
      ```

3. Create a virtual environment in which to run the app.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install python dependencies.

```bash
pip3 install -U -r requirements.txt
```
  - This program was tested in python 3.8+ environments.


5. Run the app.

```python
python3 -m streamlit run Extraction_app.py
```

### Update app

```bash
git pull
source .venv/bin/activate
opam update && opam upgrade grewpy
pip3 install -U -r requirements.txt
```
### Troubleshooting 

If opam 2 in not available in your favorite package manager, you should be able to install version 2.0.6 with the following commands:

```bash
wget -q https://github.com/ocaml/opam/releases/download/2.0.6/opam-2.0.6-x86_64-linux
sudo mv opam-2.0.6-x86_64-linux /usr/local/bin/opam
sudo chmod a+x /usr/local/bin/opam
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

Thanks to Sylvain, Bruno and Guy.
