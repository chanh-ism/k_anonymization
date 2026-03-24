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

# Test


For example, notice the cell below contains the `hide-input` tag:

```{code-cell} ipython3
---
tags: [hide-input]
editable: true
slideshow:
  slide_type: ''
---
print(2+2)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

More thigns

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(2+2+2)
```

```{code-cell} ipython3
---
tags: [remove-cell]
editable: true
slideshow:
  slide_type: ''
---
# %cd ..
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from k_anonymization import datasets
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from k_anonymization.utils.data_table import show
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
show(datasets.ADULT.df, str(datasets.ADULT))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
datasets.ADULT.describe()
```
