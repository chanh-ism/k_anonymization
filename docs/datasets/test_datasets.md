---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Test Datasets

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from k_anonymization import datasets
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Mini Crime

Adult dataset is the de facto dataset for evaluation in most research papers.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
datasets.MINI_CRIME.describe()
```

```{code-cell} ipython3
datasets.MINI_CRIME.hierarchies.generalization_trees
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
datasets.MINI_CRIME
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Mini Patient

Adult dataset is the de facto dataset for evaluation in most research papers.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
datasets.MINI_PATIENT.describe()
```

```{code-cell} ipython3
datasets.MINI_PATIENT.hierarchies.generalization_trees
```

```{code-cell} ipython3
datasets.MINI_PATIENT
```
