# Recreating the datasets

The large generated `.h5` files are derived artifacts, not source data. They
should not be committed. To reproduce an experiment, install the data-generation
dependencies, download a dataset from its original publisher, and create the HDF5
file locally:

```bash
source ~/.venvs/nether/bin/activate
python -m pip install -e ".[data-generation]"
```

Run all commands below from the repository root. Keep downloaded archives and
extracted source data outside the repository when possible. This avoids
redistributing data under the wrong licence and keeps the Git history small.

## AWA2 / Animals (supported workflow)

This is the dataset required by `awa2_problem_1.ipynb`.

1. Download the base package and image package from the official
   [Animals with Attributes 2 page](https://cvml.ista.ac.at/AwA2/). Read the
   dataset terms and the per-image licence information on that page.
2. Extract both packages into one directory. The relevant part should look like:

   ```text
   Animals_with_Attributes2/
   ├── classes.txt
   ├── predicates.txt
   ├── predicate-matrix-binary.txt
   └── JPEGImages/
       ├── antelope/
       │   └── ...jpg
       └── ...
   ```

3. Convert the source data. In WSL, a download extracted on Windows might be at
   `/mnt/c/Users/<username>/Downloads/Animals_with_Attributes2`:

   ```bash
   python -m nether.Data.data_creation awa2 \
     --source /mnt/c/Users/<username>/Downloads/Animals_with_Attributes2
   ```

   The command center-crops and resizes each image to 64 x 64 RGB, attaches the
   85 class-level attributes, writes 5,000-row chunks to `Data/animals/`, and
   verifies the result. It refuses to overwrite existing chunks unless `--force`
   is supplied.

4. Verify an existing local build without recreating it:

   ```bash
   python -m nether.Data.data_creation awa2 --verify-only
   ```

5. Start the example only after verification succeeds:

   ```bash
   jupyter lab awa2_problem_1.ipynb
   ```

The official release currently produces eight files named
`full_data_0_64.h5` through `full_data_7_64.h5`, containing 37,322 rows in
total. The notebook loader then selects at most 100 images per class, yielding
the 5,000-instance experimental subset.

## All dataset commands

Every paper dataset now has a separate subcommand. Each command validates its
source layout, refuses accidental overwrites, creates the loader-facing HDF5
schema, and verifies the completed output. Use `--force` only when intentionally
replacing an existing artifact.

| Command | Official source | Generated artifact | Verified shape |
| --- | --- | --- | --- |
| `awa2` | [Animals with Attributes 2](https://cvml.ista.ac.at/AwA2/) | `Data/animals/full_data_*_64.h5` | 37,322 rows, 8 chunks |
| `housing` | [Austin Housing Prices](https://www.kaggle.com/datasets/ericpierce/austinhousingprices) | `Data/housing/housing_small.h5` | 5,508 × 12,327 |
| `music` | [Free Music Archive](https://github.com/mdeff/fma) | `Data/music/music.h5` | 9,217 × 532 |
| `birds` | [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) | `Data/birds/full_data.h5` | 11,788 × 12,600 |
| `objects` | [PASCAL VOC](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/) | `Data/objects/full_data.h5` | 17,125 × 12,308 |
| `faces` | [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | `Data/faces/full_data.h5` | 10,000 × 12,328 |
| `medical` | [Derm7pt](https://derm.cs.sfu.ca/) | `Data/medical/full_data.h5` | 1,011 × 12,301 |
| `light` | [StarLightCurves](https://www.timeseriesclassification.com/description.php?Dataset=StarLightCurves) | `Data/light/full_data.h5` | 9,236 × 1,045 |
| `motifs` | [NATOPS through aeon](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.datasets.load_classification.html) | `Data/motifs/full_data.h5` | 5,000 × 690 |

### Housing

Extract the Kaggle download so the CSV and images share one directory:

```text
austin-housing/
├── austinHousingData.csv
└── homeImages/
```

```bash
python -m nether.Data.data_creation housing --source /path/to/austin-housing
```

The builder encodes the three string-valued columns, removes identifiers and
free text, and center-crops each source image to 64 × 64 RGB.

### Music

Download and extract `fma_metadata.zip`. No MP3 files are needed. Point the
command at the directory containing these four files:

```text
fma_metadata/
├── tracks.csv
├── genres.csv
├── features.csv
└── echonest.csv
```

```bash
python -m nether.Data.data_creation music --source /path/to/fma_metadata
```

### Birds

Extract CUB-200-2011 and pass either its parent or the `CUB_200_2011` directory:

```text
CUB_200_2011/
├── images.txt
├── attributes.txt
├── attributes/image_attribute_labels.txt
└── images/
```

```bash
python -m nether.Data.data_creation birds --source /path/to/CUB_200_2011
```

The 312 image-level attributes are followed by center-cropped 64 × 64 RGB data.

### Objects

Download the PASCAL VOC 2012 train/validation data and development kit. The
builder accepts either `VOC2012` itself or a parent containing
`VOCdevkit/VOC2012`:

```text
VOCdevkit/VOC2012/
├── JPEGImages/
└── Annotations/
```

```bash
python -m nether.Data.data_creation objects --source /path/to/VOCdevkit/VOC2012
```

It creates the canonical 20-class multi-label representation and 64 × 64 RGB
subsymbolic features.

### Faces

Accept the CelebA download agreement, then download the aligned and cropped
images (`img_align_celeba.zip`) and attribute annotations
(`list_attr_celeba.txt`). Extract them as:

```text
celeba/
├── list_attr_celeba.txt
└── img_align_celeba/
    ├── 000001.jpg
    └── ...
```

```bash
python -m nether.Data.data_creation faces --source /path/to/celeba
```

This intentionally uses annotation rows `000001.jpg` through `010000.jpg`: the
first 10,000 images in the official ordering. The 40 `-1`/`1` attributes are
converted to `0`/`1`; images are center-cropped and resized to 64 × 64 RGB using
the same subsymbolic preprocessing as AWA2.

### Medical

Download the Derm7pt release. The command accepts the released layout or a
flattened equivalent:

```text
release_v0/
├── meta/
│   ├── meta.csv
│   ├── train_indexes.csv
│   ├── valid_indexes.csv
│   └── test_indexes.csv
└── images/
```

Create the normal 64 × 64 dataset:

```bash
python -m nether.Data.data_creation medical --source /path/to/release_v0
```

The diagnosis experiments use paired clinical and dermoscopy images at
512 × 512. Their generated matrix is very large (about 1.6 GB per HDF5 file):

```bash
python -m nether.Data.data_creation medical \
  --variant diagnosis --source /path/to/release_v0
```

The source implementation lives in
`Data/medical/create_diagnosis_data.py`. The learned-concept artifact is a
second-stage model output produced by
`Data/medical/create_diagnosis_data_with_predicted_concepts.py`; it is not a
source-data conversion and should be reported with its model seed/checkpoint.

### Light

Download the StarLightCurves train and test text files into one directory:

```text
StarLightCurves/
├── StarLightCurves_TRAIN.txt
└── StarLightCurves_TEST.txt
```

```bash
python -m nether.Data.data_creation light --source /path/to/StarLightCurves
```

The builder combines both official splits, derives seven symbolic features in
each of three windows, and keeps the 1,024 time-series values.

### Motifs

This synthetic dataset downloads NATOPS through `aeon`, creates shifted classes,
and concatenates ten sampled motifs per row. Both NumPy and Python sampling use
the recorded seed:

```bash
python -m nether.Data.data_creation motifs --seed 0
```

## Verification

Verify one artifact without rebuilding it:

```bash
python -m nether.Data.data_creation faces --verify-only
python -m nether.Data.data_creation medical --variant diagnosis --verify-only
```

Or verify every expected artifact in one metadata-only pass:

```bash
python -m nether.Data.data_creation verify-all
```

Verification reads HDF5 metadata rather than loading the complete matrices. It
checks the `full_data` key, expected row count, symbolic width, and subsymbolic
dimensions. Dataset licences still apply to local source and generated files;
none of the commands grants redistribution rights.
