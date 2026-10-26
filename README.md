# NeTheR

NeTheR revises an imperfect propositional theory by greedily inserting or removing
symbolic literals and by inventing neural concepts learned from subsymbolic data.
This repository accompanies the paper *Propositional Theory Revision through Neural
Concept Invention*.

## Repository contents

All experimental results reported for the project are stored in
[`Results/`](Results/). The experiment scripts and the generators used to create
the paper's tables and figures are stored in [`Experiments/`](Experiments/).

## What NeTheR does

NeTheR starts with an imperfect propositional theory, labeled examples, and
subsymbolic inputs such as images. It searches for a small revision that improves
the theory. When the available symbolic attributes are insufficient, the revision
may contain a neural predicate learned directly from the subsymbolic data.

The Animals notebook contains the following illustrative examples. Each row also
contains a 64 × 64 RGB image, omitted from the table for brevity. The `newworld`
column is shown to explain the generated labels, but it is deliberately removed
from the expert theory and is not available for symbolic revision.

| Animal | `oldworld` | `newworld` | `strainteeth` | `furry` | Label | Expert prediction | Revised prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ox | 1 | 0 | 0 | 1 | 1 | 0 | 1 |
| mouse | 1 | 0 | 0 | 1 | 1 | 0 | 1 |
| lion | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| bobcat | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| dalmatian | 0 | 1 | 1 | 1 | 0 | 0 | 0 |

The imperfect expert theory is:

```text
M' = oldworld = 1
     AND strainteeth = 0
     AND furry = 0
```

It recognizes the lion, but incorrectly rejects the positive ox and mouse
examples. Their shared explanation depends on the missing `newworld = 0`
condition. The ground-truth theory used to generate the labels is:

```text
M* = oldworld = 1
     AND (newworld = 0
          OR (strainteeth = 0 AND furry = 0))
```

NeTheR is not given this ground-truth theory. From the labels and image pixels it
can instead invent a neural predicate, written below as `NN(image)`, that captures
the missing visual concept. The successful repair has the conceptual form:

```text
M_NeTheR = oldworld = 1
           AND (NN(image)
                OR (strainteeth = 0 AND furry = 0))
```

Here `NN(image)` learns to cover examples represented in the ground truth by the
unavailable `newworld = 0` branch. The result remains a logical theory, but one of
its predicates is grounded in the image rather than a supplied symbolic column.
The exact neural-predicate identifier, learned weights, revision path, and scores
can vary between runs; the notebook displays the concrete result of the current
run.

## Quick start

NeTheR supports Python 3.10-3.12. The commands below create an isolated environment,
install the package and Jupyter, and launch the reproducible AWA2 example.

```bash
python3 -m venv ~/.venvs/nether
source ~/.venvs/nether/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab awa2_problem_1.ipynb
```

In PyCharm, select `/home/<username>/.venvs/nether/bin/python` as the project
interpreter and choose **NeTheR (WSL)** as the notebook kernel. To register that
kernel manually after activating the environment, run:

```bash
python -m ipykernel install --user --name nether --display-name "NeTheR (WSL)"
```

The default Linux PyTorch wheel includes CUDA 12.1 support. For a smaller CPU-only
installation, install the CPU build before installing NeTheR:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

NeTheR automatically selects CUDA when it is available and otherwise uses the CPU.


## AWA2 example

The notebook `awa2_problem_1.ipynb` reproduces the first saved Animals
problem (`2_2_1_0_expert.json`). It:

1. loads the prepared 5,000-instance AWA2 subset and displays images and attributes;
2. shows the expert theory and ground-truth theory;
3. reports the expert theory's baseline F1 score;
4. runs NeTheR with `edit_distance=5`, `k=3`, `freeze=True`,
   `symb_allowed=True`, and `epochs=75`;
5. evaluates and displays the revised theory on the held-out test split.

The source data is [Animals with Attributes 2](https://cvml.ista.ac.at/AwA2/),
which contains 37,322 images from 50 animal classes and 85 class attributes. This
repository's prepared subset takes up to 100 images per class and stores resized
64 x 64 RGB pixels together with the attributes. Consult the original dataset page
for its citation and per-image licensing information.

The generated HDF5 files are intentionally not part of the repository. Before
opening the notebook, follow the download, conversion, and verification steps in
[`Data/README.md`](Data/README.md). The supported path is:

```bash
python -m nether.Data.data_creation awa2 \
  --source /path/to/Animals_with_Attributes2
python -m nether.Data.data_creation awa2 --verify-only
jupyter lab awa2_problem_1.ipynb
```

The same data guide gives one creation command for every dataset used in the
paper. To check all locally generated artifacts without loading their full
matrices, run `python -m nether.Data.data_creation verify-all`.

The saved problem uses the paper's 80%/10%/10% train/validation/test split. The
notebook intentionally uses 75 maximum epochs, as requested for this example; the
paper's main evaluation reports 50 maximum epochs. Early stopping uses patience 5.

## Installation choices

The core package contains only dependencies required to load saved problems and run
NeTheR. Optional tools are grouped as extras:

```bash
# Notebook support
python -m pip install -e ".[notebook]"

# Dataset/problem generation (includes Weka bindings)
python -m pip install -e ".[data-generation]"

# Critical-difference plots used in the paper
python -m pip install -e ".[analysis]"
```

`requirements.txt` is the supported end-user entry point and installs the editable
package with notebook support. Machine-specific Conda export paths are intentionally
not part of the installation instructions.
