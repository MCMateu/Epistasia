# Epistasia

**Epistasia** is a Python library to analyze **binary landscapes**
(genotype–phenotype or community–function maps) and quantify epistatic
interactions across orders.

📘 Documentation: https://MCMateu.github.io/epistasia/  
💻 Source code: https://github.com/MCMateu/epistasia

---

## Installation

### From GitHub (recommended while the API is evolving)

```bash
pip install "epistasia @ git+https://github.com/MCMateu/epistasia.git"
```

### Editable install (development)

```bash
git clone https://github.com/MCMateu/epistasia.git
cd epistasia
pip install -e .
```

## Quick start

```python
import epistasia as ep

# Expected columns: x1..xN, F (optionally: replicate)
L = ep.landscape_from_file("my_landscape.csv")

out = ep.walsh_analysis(L)

print(out)
```

## Status

Epistasia is under active development.

