"""Create and verify the derived datasets used by NeTheR.

Command-line workflow
---------------------
The AWA2/Animals dataset used by ``awa2_problem_1.ipynb`` can be recreated with::

    python -m nether.Data.data_creation awa2 \
        --source /path/to/Animals_with_Attributes2

Run the command from the repository root with the NeTheR environment activated.
Every other paper dataset has its own subcommand following the same
``--source`` / ``--verify-only`` pattern. Run ``verify-all`` for a metadata-only
check of all generated artifacts. See ``Data/README.md`` for downloads, source
layouts, licensing notes, and exact commands.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _awa2_root(source: Path) -> Path:
    """Return the directory containing AWA2 metadata and ``JPEGImages``."""
    source = source.expanduser().resolve()
    nested = source / "Animals_with_Attributes2"
    if nested.is_dir():
        source = nested

    required = [
        source / "JPEGImages",
        source / "classes.txt",
        source / "predicates.txt",
        source / "predicate-matrix-binary.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"AWA2 source is incomplete. Missing:\n  - {formatted}")
    return source


def _read_indexed_names(path: Path) -> list[str]:
    names: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            _, name = line.rstrip("\n").split(maxsplit=1)
            names.append(name)
    return names


def _pixel_columns(size: int) -> list[str]:
    channels = ("r", "g", "b")
    return [
        f"{channels[channel]}_{row}_{column}"
        for row in range(size)
        for column in range(size)
        for channel in range(3)
    ]


def _prepare_image(path: Path, size: int):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        edge = min(width, height)
        left = (width - edge) // 2
        top = (height - edge) // 2
        image = image.crop((left, top, left + edge, top + edge))
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).reshape(-1)


def create_awa2(source: Path, output: Path, chunk_size: int, image_size: int, force: bool) -> None:
    """Convert the official AWA2 release into chunked HDF5 files."""
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    source = _awa2_root(source)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    classes = _read_indexed_names(source / "classes.txt")
    predicates = _read_indexed_names(source / "predicates.txt")
    matrix = np.loadtxt(source / "predicate-matrix-binary.txt", dtype=np.uint8)
    if matrix.shape != (len(classes), len(predicates)):
        raise ValueError(
            "AWA2 predicate matrix shape does not match classes/predicates: "
            f"{matrix.shape} != ({len(classes)}, {len(predicates)})"
        )
    attributes_by_class = dict(zip(classes, matrix, strict=True))

    image_paths = sorted(
        path
        for path in (source / "JPEGImages").rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No JPEG images found below {source / 'JPEGImages'}")

    output_files = [
        output / f"full_data_{index}_64.h5"
        for index in range((len(image_paths) + chunk_size - 1) // chunk_size)
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"{existing[0]} already exists. Use --force to replace generated HDF5 files."
        )

    pixel_columns = _pixel_columns(image_size)
    for index, start in enumerate(range(0, len(image_paths), chunk_size)):
        chunk = image_paths[start:start + chunk_size]
        animal_names = [path.parent.name for path in chunk]
        unknown = sorted(set(animal_names) - attributes_by_class.keys())
        if unknown:
            raise ValueError(f"Image directories not listed in classes.txt: {unknown}")

        pixels = np.vstack([
            _prepare_image(path, image_size)
            for path in tqdm(chunk, desc=f"AWA2 chunk {index + 1}/{len(output_files)}")
        ])
        labels = pd.DataFrame(
            [attributes_by_class[name] for name in animal_names],
            columns=predicates,
            dtype=np.uint8,
        )
        frame = pd.concat(
            [pd.Series(animal_names, name="animal"), labels,
             pd.DataFrame(pixels, columns=pixel_columns)],
            axis=1,
        )
        frame.to_hdf(output_files[index], key="full_data", mode="w")

    for filename in ("classes.txt", "predicates.txt", "predicate-matrix-binary.txt"):
        destination = output / filename
        if force or not destination.exists():
            shutil.copyfile(source / filename, destination)

    verify_awa2(output, expected_rows=len(image_paths), image_size=image_size)


def verify_awa2(output: Path, expected_rows: int | None = 37_322, image_size: int = 64) -> None:
    """Check that generated AWA2 chunks have the schema expected by the loader."""
    import pandas as pd

    output = output.expanduser().resolve()
    files = sorted(output.glob("full_data_*_64.h5"))
    if not files:
        raise FileNotFoundError(f"No full_data_*_64.h5 files found in {output}")

    predicates = _read_indexed_names(output / "predicates.txt")
    expected_columns = ["animal", *predicates, *_pixel_columns(image_size)]
    rows = 0
    for path in files:
        frame = pd.read_hdf(path, key="full_data")
        if list(frame.columns) != expected_columns:
            raise ValueError(f"Unexpected columns in {path}")
        rows += len(frame)

    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} AWA2 rows, found {rows:,}")
    print(f"Verified {rows:,} AWA2 rows in {len(files)} HDF5 files at {output}")


def _main() -> None:
    from nether.Data.dataset_builders import (
        create_birds,
        create_faces,
        create_housing,
        create_light,
        create_motifs,
        create_music,
        create_objects,
        verify_dataset,
    )
    from nether.Data.medical.create_diagnosis_data import create_base, create_diagnosis

    repository_data = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create or verify locally generated NeTheR datasets.",
        epilog="See Data/README.md for downloads, expected layouts, and dataset status.",
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    awa2 = subparsers.add_parser("awa2", help="create or verify the Animals with Attributes 2 data")
    awa2.add_argument("--source", type=Path, help="extracted AWA2 directory")
    awa2.add_argument("--output", type=Path, default=repository_data / "animals")
    awa2.add_argument("--chunk-size", type=int, default=5_000)
    awa2.add_argument("--image-size", type=int, default=64)
    awa2.add_argument("--force", action="store_true", help="replace existing generated chunks")
    awa2.add_argument("--verify-only", action="store_true", help="check existing output without rebuilding")

    dataset_defaults = {
        "housing": ("housing", "housing_small.h5"),
        "music": ("music", "music.h5"),
        "birds": ("birds", "full_data.h5"),
        "objects": ("objects", "full_data.h5"),
        "faces": ("faces", "full_data.h5"),
        "light": ("light", "full_data.h5"),
    }
    builders = {
        "housing": create_housing,
        "music": create_music,
        "birds": create_birds,
        "objects": create_objects,
        "faces": create_faces,
        "light": create_light,
    }
    for name, (directory, _) in dataset_defaults.items():
        command = subparsers.add_parser(name, help=f"create or verify the {name} dataset")
        command.add_argument("--source", type=Path, help="downloaded/extracted source directory")
        command.add_argument("--output", type=Path, default=repository_data / directory)
        command.add_argument("--force", action="store_true", help="replace an existing generated file")
        command.add_argument("--verify-only", action="store_true", help="check existing output without rebuilding")

    motifs = subparsers.add_parser("motifs", help="create or verify the synthetic NATOPS motif dataset")
    motifs.add_argument("--output", type=Path, default=repository_data / "motifs")
    motifs.add_argument("--seed", type=int, default=0)
    motifs.add_argument("--force", action="store_true")
    motifs.add_argument("--verify-only", action="store_true")

    medical = subparsers.add_parser("medical", help="create or verify the Derm7pt datasets")
    medical.add_argument("--source", type=Path, help="extracted Derm7pt release directory")
    medical.add_argument("--output", type=Path, default=repository_data / "medical")
    medical.add_argument("--variant", choices=("base", "diagnosis"), default="base")
    medical.add_argument("--force", action="store_true")
    medical.add_argument("--verify-only", action="store_true")

    subparsers.add_parser("verify-all", help="verify every generated dataset currently in Data/")
    args = parser.parse_args()

    if args.dataset == "verify-all":
        verify_awa2(repository_data / "animals")
        for name, (directory, filename) in dataset_defaults.items():
            verify_dataset(name, repository_data / directory / filename)
        verify_dataset("motifs", repository_data / "motifs" / "full_data.h5")
        verify_dataset("medical", repository_data / "medical" / "full_data.h5")
        verify_dataset("medical-diagnosis", repository_data / "medical" / "full_data_diagnosis.h5")
        return

    if args.dataset == "awa2":
        if args.verify_only:
            verify_awa2(args.output, image_size=args.image_size)
        elif args.source is None:
            parser.error("awa2 requires --source unless --verify-only is used")
        else:
            create_awa2(args.source, args.output, args.chunk_size, args.image_size, args.force)
        return

    if args.dataset in builders:
        _, default_filename = dataset_defaults[args.dataset]
        verify_path = args.output if args.output.suffix.lower() == ".h5" else args.output / default_filename
        if args.verify_only:
            verify_dataset(args.dataset, verify_path)
        elif args.source is None:
            parser.error(f"{args.dataset} requires --source unless --verify-only is used")
        else:
            builders[args.dataset](args.source, args.output, args.force)
        return


    if args.dataset == "motifs":
        verify_path = args.output if args.output.suffix.lower() == ".h5" else args.output / "full_data.h5"
        if args.verify_only:
            verify_dataset("motifs", verify_path)
        else:
            create_motifs(args.output, args.force, args.seed)
        return

    if args.dataset == "medical":
        filename = "full_data.h5" if args.variant == "base" else "full_data_diagnosis.h5"
        verify_path = args.output if args.output.suffix.lower() == ".h5" else args.output / filename
        verifier = "medical" if args.variant == "base" else "medical-diagnosis"
        if args.verify_only:
            verify_dataset(verifier, verify_path)
        elif args.source is None:
            parser.error("medical requires --source unless --verify-only is used")
        elif args.variant == "base":
            destination = create_base(args.source, args.output, args.force)
            verify_dataset("medical", destination)
        else:
            _, destination = create_diagnosis(args.source, args.output, args.force)
            verify_dataset("medical-diagnosis", destination)


if __name__ == "__main__":
    _main()
    raise SystemExit(0)


LEGACY_RECIPE = r'''
# ---------------------------------------------------------------------------
# Legacy research recipes (kept for provenance; not reached by the CLI above)
# ---------------------------------------------------------------------------
import ast
import os
from ast import parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from PIL import Image
def make_image_square(image):
    # Get dimensions
    width, height = image.size
    if width == height:
        return image  # Already square

    # Calculate the size of the new square image
    new_size = min(width, height)

    # Crop the image to make it square
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2

    return image.crop((left, top, right, bottom))


def repeat_edge(image, target_size):
    width, height = image.size
    while width < target_size or height < target_size:
        if width < target_size:
            right_edge = image.crop((width - 3, 0, width, height))
            image = Image.new(image.mode, (width + 3, height))
            image.paste(right_edge, (width, 0))
            width += 3

        if height < target_size:
            bottom_edge = image.crop((0, height - 3, width, height))
            image = Image.new(image.mode, (width, height + 3))
            image.paste(bottom_edge, (0, height))
            height += 3

    return image


def process_image(input_path):
    # Open image
    image = Image.open(input_path)

    # Make the image square
    image = make_image_square(image)

    # Repeat edges if the image is too small
    if image.size[0] < 64 or image.size[1] < 64:
        image = repeat_edge(image, 64)

    # Resize the image to 128x128
    image = image.resize((64, 64), Image.LANCZOS)

    return image


def list_all_files(directory):
    # List to store all file paths
    file_paths = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths


from warnings import simplefilter
#------------------ ANIMALS ----------------#
#Download the source files at https://cvml.ista.ac.at/AwA2/
categorical_columns = []
with open("data/animals/predicates.txt") as f:
    lines = f.readlines()
for line in lines:
    splitted = line.split("\t")
    categorical_columns.append(splitted[1][:-1])
droppable_cols = categorical_columns.copy()
classes = []
with open("data/animals/classes.txt") as f:
    lines = f.readlines()
for line in lines:
    splitted = line.split("\t")
    classes.append(splitted[1][:-1])
with open("data/animals/predicate-matrix-binary.txt") as f:
    lines = f.readlines()
all_preds = []
for line in lines:
    all_preds.append([int(x) for x in line[:-1].split(" ")])

for it in range(0, 8):
    new_df = [[] for x in range(64 * 64 * 3)]
    class_list = []
    for img in tqdm(list_all_files("data/animals/JPEGImages/")[it * 5000:(it + 1) * 5000]):
        class_list.append(img[img.find("\\") + 1:img.find("_")])
        image = process_image(img)
        im_array = np.asarray(image.convert('RGB'))
        counter = 0
        for i in range(64):
            for x in range(64):
                for c in range(3):
                    new_df[counter].append(im_array[i][x][c])
                    counter += 1
    animal_data = pd.DataFrame(columns=["animal"] + all_preds)
    for animal in tqdm(class_list):
        animal_id = classes.index(animal) - 1
        animal_data.loc[len(animal_data.index)] = [animal] + all_preds[animal_id]
    counter = 0
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    for i in tqdm(range(64)):
        for x in range(64):
            for c in range(3):
                animal_data[["r", "g", "b"][c] + "_" + str(i) + "_" + str(x)] = new_df[counter]
                counter += 1
    animal_data.to_hdf("data/animals/full_data_" + str(it) + "_64.h5", key='full_data', mode='w', )

#
# #----------------------HOUSING-------------------------
# Download the source files at https://www.kaggle.com/datasets/ericpierce/austinhousingprices
housing_data=pd.read_csv("data/housing/austinHousingData.csv")
housing_data=housing_data.drop(columns=["zpid","description","streetAddress","zipcode","latitude","longitude","latest_saledate"])
housing_data.city=housing_data.city.astype('category').cat.codes
housing_data.homeType=housing_data.homeType.astype('category').cat.codes
housing_data.latestPriceSource=housing_data.latestPriceSource.astype('category').cat.codes
categorical_columns=housing_data.drop(columns=["homeImage"]).columns
droppable_cols=['hasCooling', 'hasGarage', 'hasHeating', 'hasSpa', 'hasView',
       'homeType', 'parkingSpaces', 'yearBuilt', 'latestPrice','numOfParkingFeatures', 'numOfPatioAndPorchFeatures',
       'numOfSecurityFeatures', 'numOfWaterfrontFeatures',
       'numOfWindowFeatures','lotSizeSqFt',
       'livingAreaSqFt','numOfBathrooms', 'numOfBedrooms', 'numOfStories']
new_df = [[] for x in range(64 * 64 * 3)]

for img in tqdm(list(housing_data["homeImage"])):
    image = process_image("data/housing/homeImages/" + img)
    im_array = np.asarray(image.convert('RGB'))
    counter = 0
    for i in range(64):
        for x in range(64):
            for c in range(3):
                new_df[counter].append(im_array[i][x][c])
                counter += 1
housing_data = housing_data.drop(columns=["homeImage"])
counter = 0
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
for i in tqdm(range(64)):
    for x in range(64):
        for c in range(3):
            housing_data[["r", "g", "b"][c] + "_" + str(i) + "_" + str(x)] = new_df[counter]
            counter += 1
housing_data.to_hdf("data/housing/housing_small.h5", key='full_data', mode='w', )

# #------------------------------MUSIC------------------------------
#Download the sourcefiles at https://github.com/mdeff/fma
tracks = pd.read_csv('Data/music/tracks.csv', index_col=0, header=[0, 1])
COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
           ('track', 'genres'), ('track', 'genres_all')]
for column in COLUMNS:
    tracks[column] = tracks[column].map(ast.literal_eval)

COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
           ('album', 'date_created'), ('album', 'date_released'),
           ('artist', 'date_created'), ('artist', 'active_year_begin'),
           ('artist', 'active_year_end')]
for column in COLUMNS:
    tracks[column] = pd.to_datetime(tracks[column])

SUBSETS = ('small', 'medium', 'large')
try:
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
        'category', categories=SUBSETS, ordered=True)
except (ValueError, TypeError):
    # the categories and ordered arguments were removed in pandas 0.25
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
        pd.CategoricalDtype(categories=SUBSETS, ordered=True))

COLUMNS = [('track', 'genre_top'), ('track', 'license'),
           ('album', 'type'), ('album', 'information'),
           ('artist', 'bio')]
for column in COLUMNS:
    tracks[column] = tracks[column].astype('category')
genres = pd.read_csv('Data/music/genres.csv', index_col=0)
features = pd.read_csv('Data/music/features.csv', index_col=0, header=[0, 1, 2])
echonest = pd.read_csv('Data/music/echonest.csv', index_col=0, header=[0, 1, 2])

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks.shape, genres.shape, features.shape, echonest.shape
large = tracks[tracks['set', 'subset'] <= 'large']
small_tracks = large['track'][['duration', 'genre_top']]
small_indices = list(small_tracks.index)

echonest_indices = echonest.index
useful_indices = echonest_indices.intersection(small_indices)
useful_tracks = small_tracks.loc[useful_indices]

# Map each genre to its ID from the genre table
useful_tracks["genre_top"] = useful_tracks["genre_top"].transform(
    lambda x: genres.index[genres['title'] == x].tolist()[0])

# Discretisize the duration per 30s
useful_tracks["duration"] = pd.cut(useful_tracks["duration"],
                                   bins=list(range(0, 1001, 30)),
                                   labels=list(range(0, len(range(0, 1001, 30)) - 1)))
useful_audio_features = features.loc[useful_indices]
echonest_features = echonest.loc[useful_indices]
useful_echonest_features = echonest_features['echonest'][['audio_features', 'social_features']]
discretized_echonest = pd.DataFrame()
for i in useful_echonest_features.columns:
    # discretisize tempo per 25bpm
    if i[1] == 'tempo':
        discretized_echonest[i[1]] = pd.cut(useful_echonest_features[i[0]][i[1]],
                                            bins=list(range(0, 286, 25)),
                                            labels=list(range(0, len(range(0, 286, 25)) - 1)))
    else:
        # discretize all echonest features per 0.1
        discretized_echonest[i[1]] = pd.cut(useful_echonest_features[i[0]][i[1]],
                                            bins=[-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                            labels=list([x for x in range(0, 10)]))
# Add track_data, echonest features and audio features together
full_data = pd.concat([useful_tracks, discretized_echonest, useful_audio_features], axis=1).dropna()
full_data.columns = full_data.columns.astype(str)
# Normalize the continous data (discretization of echonest and track data happens above)
numeric_columns = list(full_data.select_dtypes('float64').columns)
categorical_columns = list(set(full_data.columns) - set(numeric_columns))
pipeline = ColumnTransformer([('num', StandardScaler(), numeric_columns)])

new_X = np.concatenate((np.array(full_data[categorical_columns]), pipeline.fit_transform(full_data)), axis=1)
full_data = pd.DataFrame(data=new_X, index=full_data.index, columns=categorical_columns + numeric_columns).drop(
    columns="song_currency")
categorical_columns = list(set(full_data.columns) - set(numeric_columns))
droppable_cols = categorical_columns


# #------------------BIRDS ----------------------------"
#Download source files at https://www.vision.caltech.edu/datasets/cub_200_2011/
id2img = {}
img2id = {}
with open("data/birds/images.txt") as f:
    lines = f.readlines()
    for line in lines:
        splitted = line.split(" ")
        id = int(splitted[0])
        img = splitted[1].strip()
        id2img[id] = img
        img2id[img] = id
classes = {}
with open("data/birds/classes.txt") as f:
    lines = f.readlines()
    for line in lines:
        splitted = line.split(" ")
        id = int(splitted[0])
        descr = splitted[1].strip()
        classes[id] = descr
id2class = {}
with open("data/birds/image_class_labels.txt") as f:
    lines = f.readlines()
    for line in lines:
        splitted = line.split(" ")
        id = int(splitted[0])
        class_id = int(splitted[1])
        id2class[id] = class_id
attributes = {}
with open("data/birds/attributes.txt") as f:
    lines = f.readlines()
    for line in lines:
        splitted = line.split(" ")
        id = int(splitted[0])
        descr = splitted[1].strip()
        attributes[id] = descr

all_dicts = []
with open("data/birds/attributes/image_attribute_labels.txt") as f:
    lines = f.readlines()
    new_row = {}
    for line in lines:
        splitted = line.split(" ")
        id = int(splitted[0])
        att_id = int(splitted[1])
        truth = int(splitted[2])
        certainty = int(splitted[3])
        new_row[attributes[att_id]] = truth * (certainty > 1)
        if len(new_row) == 312:
            all_dicts.append(new_row)
            new_row = {}

full_data = pd.DataFrame.from_records(all_dicts)
new_df = [[] for x in range(64 * 64 * 3)]
todo = [(id, id2img[id], classes[id2class[id]]) for id in id2img]
for item in tqdm(todo):
    img = "data/birds/images/" + item[1]
    image = process_image(img)
    im_array = np.asarray(image.convert('RGB'))
    counter = 0
    for i in range(64):
        for x in range(64):
            for c in range(3):
                new_df[counter].append(im_array[i][x][c])
                counter += 1
counter = 0
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
for i in tqdm(range(64)):
    for x in range(64):
        for c in range(3):
            full_data[["r", "g", "b"][c] + "_" + str(i) + "_" + str(x)] = new_df[counter]
            counter += 1
cat_cols = list(attributes.values())
sorted_colours = sorted(list(set(full_data.columns) - set(cat_cols)),
                        key=lambda x: (-ord(x[0]), int(x[2:x.find("_", 2)]), int(x[x.find("_", 2) + 1:])))
final_data = full_data[list(cat_cols) + sorted_colours].reset_index(drop=True)
final_data.to_hdf("data/birds/full_data.h5", key='full_data', mode='w', )
#
# #---------------OBJECTS -----------------
# Download the source files at https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/
all_possible_attributes = set()
im2att = {}
for img in tqdm(os.listdir("data/objects/JPEGImages")):
    xml = img.replace(".jpg", ".xml")
    with open("data/objects/Annotations/" + xml, 'r', encoding='utf-8') as file:
        my_xml = file.read()
    dict = xmltodict.parse(my_xml)["annotation"]["object"]
    if type(dict) != list:
        dict = [dict]
    attributes = set([x["name"] for x in dict])
    im2att[img] = list(attributes)
    all_possible_attributes = all_possible_attributes.union(attributes)
cat_data=[]
cat_cols=list(all_possible_attributes)
for im in im2att:
    res={}
    data=im2att[im]
    for x in cat_cols:
        if x in data:
            res[x]=1
        else:
            res[x]=0
    cat_data.append(res)
full_data=pd.DataFrame.from_records(cat_data)
new_df=[[] for x in range(64*64*3)]

for img in tqdm(im2att):
    img="data/objects/JPEGImages/"+img
    image=process_image(img)
    im_array=np.asarray(image.convert('RGB'))
    counter=0
    for i in range(64):
        for x in range(64):
                for c in range(3):
                    new_df[counter].append(im_array[i][x][c])
                    counter+=1
counter=0
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
for i in tqdm(range(64)):
    for x in range(64):
        for c in range(3):
            full_data[["r","g","b"][c]+"_"+str(i)+"_"+str(x)]=new_df[counter]
            counter+=1

sorted_colours=sorted(list(set(full_data.columns)-set(cat_cols)), key=lambda x: (-ord(x[0]), int(x[2:x.find("_",2)]),int(x[x.find("_",2)+1:])))
final_data=full_data[list(cat_cols)+sorted_colours].reset_index(drop=True)
final_data.to_hdf("data/objects/full_data.h5",key='full_data',mode='w',)



#### Light #####
# Download the source at https://www.timeseriesclassification.com/description.php?Dataset=StarlightCurves
with open("Data/light/StarLightCurves_TRAIN.txt") as f:
    lines=f.readlines()
# with open("Data/forda/FordA_TEST.txt") as f:
#     lines+=f.readlines()
lines=lines[:250]
import math

def classify_subsequence(data):
    results = [False for _ in range(21)]  # [Peak, Drop, Hill, Valley, Flat, Plateau, Sinus Wave]
    begin=0
    window=25
    ranges=list()
    for it in range(0,3):
        if it == 2:
            extra = -window
        else:
            extra=0
        for index in range(it*math.ceil(len(data)/3),(it+1)*math.ceil(len(data)/3)+extra):

            subseq=data[index:index+window]

            # Detect peaks and valleys
            peaks, _ = find_peaks(subseq, threshold=.1)
            valleys, _ = find_peaks(-np.array(subseq), threshold=.1)

            # Sudden peak detection: a single sharp peak
            if max(subseq)-min(subseq)>2 and subseq[0]<subseq[-1]and max(subseq)>0:
                results[0+7*it]=True

            # Sudden drop detection: a single sharp drop
            if max(subseq) - min(subseq) > 2 and subseq[0] > subseq[-1] and max(subseq)<0:
                results[1+7*it] = True

            # Hill detection: significant peak relative to neighbors
            if len(peaks) > 0:
                results[2+7*it] = True

            # Valley detection: significant low point
            if len(valleys) > 0:
                results[3+7*it] = True

            # Flat detection: very low variation
            if np.std(subseq) < 0.01:  # Flat threshold
                results[4+7*it] = True

            # Plateau detection: sustained high or low values
            diffs = np.diff(subseq)
            if np.all(np.abs(diffs) < 0.03):  # Small changes over the sequence
                results[5+7*it] = True

            # Sinus wave detection: repeating peaks and valleys
            mean = sum(subseq) / len(subseq)
            if max(subseq[:3]) > .5 * mean and min(subseq[3:6]) < .5 * mean and max(subseq[6:]) > .5 * mean:
                results[6+7*it] = True
    return results


parsed=[[float(x) for x in z.split("  ") if len(x.strip())>0][1:] for z in lines]

classified=[]

all_max=max([max(x) for x in parsed])


all_max=max([max(x) for x in parsed])
for x in tqdm(parsed):
    classified.append(classify_subsequence(x))


for i in range(0,len(classified)):
    # print(classified[i])
    if classified[i][8]:
        plt.plot(range(0,len(parsed[i])), parsed[i])
plt.show()

data={}
names=["begin_peak", "begin_drop", "begin_hill", "begin_valley","begin_flat","begin_plateau","begin_sinus", "middle_peak", "middle_drop", "middle_hill", "middle_valley","middle_flat","middle_plateau","middle_sinus", "end_peak", "end_drop", "end_hill", "end_valley","end_flat","end_plateau","end_sinus"]
for i in range(0,len(names)):
    data[names[i]]=[x[i] for x in classified]
for i in range(0,len(parsed[0])):
    data["t_"+str(i)]=[x[i] for x in parsed]
full=pd.DataFrame(data)
full.to_hdf("Data/light/full_data.h5",key='full_data',mode='w',)


##### Motifs #####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
def generate_tsmd_benchmark_ts(df, g=2):
    freqs = df['label'].value_counts()

    classes = freqs.index
    classes = classes[freqs > 1]

    if len(classes) < g:
        ValueError("TODO")

    # Pick the repeating classes randomly
    repeating_classes = np.random.choice(classes, size=g, replace=False)
    # Sample one instance of every non-repeating class. Then randomly order them
    X_non_repeating = df[~df['label'].isin(repeating_classes)].copy()
    X_non_repeating = X_non_repeating.groupby('label', group_keys=False).apply(lambda x: x.sample())
    X_non_repeating = X_non_repeating.sample(frac=1).reset_index(drop=True)

    # Sample at least two motifs for each repeating class
    X_repeating = df[df['label'].isin(repeating_classes)].reset_index(drop=True)
    motifs = X_repeating.groupby('label', group_keys=False).apply(lambda x: x.sample(n=2))

    # Then complete the GT motifs by randomly sampling other motifs from repeating classes
    other_motifs = X_repeating[~X_repeating.apply(tuple, 1).isin(motifs.apply(tuple, 1))]
    other_motifs = other_motifs.sample(n=min(len(other_motifs), max(0, len(X_non_repeating) + 1 - len(motifs))),
                                       replace=False)

    all_motifs = pd.concat((motifs, other_motifs))
    all_motifs = all_motifs.sample(frac=1).reset_index(drop=True)

    gt = {c: [] for c in repeating_classes}
    ts = []

    # Concatenate instances, alternating between a non-repeating and a repeating class, until no instances left
    curr = 0
    for i in range(min(len(X_non_repeating), len(all_motifs) - 1)):
        # Motif
        motif, label, l = all_motifs.iloc[i]
        ts.append(motif)
        gt[label].append((curr, curr + l))
        curr += l

        # Non-motif
        instance, _, l = X_non_repeating.iloc[i]
        ts.append(instance)
        curr += l

    motif, label, l = all_motifs.iloc[-1]
    ts.append(motif)
    gt[label].append((curr, curr + l))
    return np.vstack(ts), gt


def convert_X_y_to_df(X, y):
    time_series = [x.T for x in X]
    lengths = [len(ts) for ts in time_series]
    df = pd.DataFrame({"ts": time_series, "label": y, "length": lengths})
    return df


def generate_tsmd_benchmark_dataset(df, N, g_min, g_max):
    # Generate time series
    benchmark_ts = []
    gts = []
    for _ in range(N):
        # Sample a number of motif sets
        g = np.random.randint(g_min, g_max + 1)

        ts, gt = generate_tsmd_benchmark_ts(df, g=g)
        benchmark_ts.append(ts)
        gts.append(gt)

    benchmark_dataset = pd.DataFrame({'ts': benchmark_ts, 'gt': gts})
    return benchmark_dataset


import os
import pandas as pd
import numpy as np


def get_g_max(n_classes):
    return int(np.floor((n_classes+1)/3))


from aeon.datasets import load_classification

# number time series per benchmark
NB_SERIES  = 250
# Percentage train
TRAIN_SIZE = 0.20

path_to_benchmark = os.path.join(".", "benchmark")
if not os.path.exists(path_to_benchmark):
    os.mkdir(path_to_benchmark)

# VAR_LENGTH_DATASETS = ["CharacterTrajectories", "SpokenArabicDigits", "JapaneseVowels"]
# FIX_LENGTH_DATASETS = [>"PenDigits", "Fungi", "NATOPS", "ECG5000"]

columns = {'ds_name': str, 'nclasses': int, 'ndim': int, 'l_min': int, 'l_max': int, 'kappa_max': int}
metadata = pd.DataFrame(columns, index=[])


def znormalize(ts):
    ts = (ts - np.mean(ts, axis=None)) / np.std(ts, axis=None)
    return ts


for ds_name in ["NATOPS"]:
    np.random.seed(0)

    # X, y
    X_train, y_train = load_classification(name=ds_name, split='train', load_equal_length=False)
    X_test, y_test = load_classification(name=ds_name, split='test', load_equal_length=False)

    df_train = convert_X_y_to_df(X_train, y_train)
    df_test = convert_X_y_to_df(X_test, y_test)
    #
    # # Combine, z-normalize, and resplit
    df = pd.concat((df_train, df_test)).reset_index(drop=True)
    classes={}
    for x in df["label"].values:
        classes[x]=list()
    print(classes)
    for index, row in df.iterrows():
        classes[row["label"]].append([x[0] for x in row["ts"]])


    ## add y-shifts to make expert knowledge usefull
    names=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W"]
    final_classes={}
    for x in classes:
        final_classes[names.pop(0)]=classes[x]
        final_classes[names.pop(0)] = [[y+2 for y in z]  for z in classes[x] ]
        final_classes[names.pop(0)] = [[y - 2 for y in z]  for z in classes[x] ]
    # for x in final_classes:
    #     plt.plot(range(0,len(final_classes[x][0])), final_classes[x][0], label=x)
    # plt.legend()
    # plt.show()
    names=list(final_classes.keys())
    final_names=[x+str(i) for i in range(0,10) for x in names]+["t"+str(i) for i in range(0,len(final_classes["A"][0])*10)]

    #
    data=[]
    for i in range(0,5000):
        ts=[]
        logical=[]
        for x in range(0,10):
            dataline = [0 for x in range(0, len(names))]
            chosen=random.randint(0,len(names)-1)
            dataline[chosen]=1

            ts+=random.sample(final_classes[names[chosen]],1)[0]
            logical+=dataline
        data.append(logical+ts)
    df=pd.DataFrame(data,columns=final_names)
    df.to_hdf("Data/motifs/full_data.h5", key='full_data', mode='w', )

##### Medical ####
#see data/medical/create_diagnosis_data.py

### faces (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) ###



'''
