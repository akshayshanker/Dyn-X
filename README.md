# DynX: Modular Dynamic Programming

⚠️ **Experimental Development Build** – APIs and documentation are incomplete.⚠️ 

DynX is a computatioanl framework for representing conjugate pairs of functional operations, a push-forward and pull-back, on Banch space as a graph. 
The focus of is on a practical rather than theoretical representation of Bellman functional recursions and push-forward measures.

The use of graphs and networks to represent dynamics is ubiquitous (see [Auclert et al (2021)](https://web.stanford.edu/~aauclert/sequence_space_jacobian.pdf) or [Stachurski and Sargent (2025)](https://networks.quantecon.org/), for instance). 
However, the critical innovation in DynX is to represent relations between **functional operations**, rather than relations between variables. 
Moreover, Unlike general graph libraries (e.g. *TensorFlow*), DynX represents **both**
computational operators **and** the functional objects that define a
recursive problem.

---

## Installation

DynX is not yet published on PyPI. You can install the latest
development build directly from GitHub:

**Main branch (bleeding-edge)**

```bash
pip install "git+https://github.com/akshayshanker/dynx.git#egg=dynx"
```

**Specific dev release (v0.18.dev4)**

```bash
pip install "git+https://github.com/akshayshanker/dynx.git@v0.18.dev4#egg=dynx"
```

To upgrade an existing installation:

```bash
pip install --upgrade --force-reinstall \
    "git+https://github.com/akshayshanker/dynx.git#egg=dynx"
```

---

## Documentation

Comprehensive documentation is in progress. 

