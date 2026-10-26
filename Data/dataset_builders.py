"""Deterministic builders and lightweight validators for NeTheR datasets."""

from __future__ import annotations

import ast
import math
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]
LIGHT_FEATURES = [
    "high_maximum", "low_minimum", "low_var", "high_var", "sinus",
    "high_mean", "big_distance",
]


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required source files are missing:\n  - " + "\n  - ".join(missing))


def _output_path(output: Path, default_name: str) -> Path:
    output = output.expanduser().resolve()
    return output / default_name if output.suffix.lower() != ".h5" else output


def _write_hdf(frame, path: Path, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it")
    frame.to_hdf(path, key="full_data", mode="w")


def _image_vector(path: Path, size: int = 64, planar: bool = True):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        edge = min(width, height)
        left, top = (width - edge) // 2, (height - edge) // 2
        image = image.crop((left, top, left + edge, top + edge))
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.uint8)
    return array.transpose(2, 0, 1).reshape(-1) if planar else array.reshape(-1)


def _pixel_columns(size: int = 64, planar: bool = True, suffix: str = "") -> list[str]:
    channels = ("r", "g", "b")
    if planar:
        return [f"{channel}_{row}_{column}{suffix}" for channel in channels
                for row in range(size) for column in range(size)]
    return [f"{channels[channel]}_{row}_{column}{suffix}" for row in range(size)
            for column in range(size) for channel in range(3)]


def _image_frame(paths: list[Path], size: int = 64, planar: bool = True):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    pixels = np.vstack([_image_vector(path, size, planar) for path in tqdm(paths, desc="Images")])
    return pd.DataFrame(pixels, columns=_pixel_columns(size, planar))


def create_faces(source: Path, output: Path, force: bool, limit: int = 10_000) -> Path:
    """Create the paper's first-10,000-image CelebA subset."""
    import pandas as pd

    source = source.expanduser().resolve()
    if (source / "celeba").is_dir():
        source = source / "celeba"
    attr_candidates = [source / "list_attr_celeba.txt", source / "Anno" / "list_attr_celeba.txt"]
    image_candidates = [source / "img_align_celeba", source / "Img" / "img_align_celeba"]
    attr_path = next((path for path in attr_candidates if path.is_file()), attr_candidates[0])
    image_dir = next((path for path in image_candidates if path.is_dir()), image_candidates[0])
    _require([attr_path, image_dir])

    with attr_path.open(encoding="utf-8") as handle:
        declared_count = int(handle.readline().strip())
        attributes = handle.readline().split()
    if attributes != CELEBA_ATTRIBUTES:
        raise ValueError("Unexpected CelebA attribute header; check the dataset version")
    labels = pd.read_csv(
        attr_path, sep=r"\s+", skiprows=2, names=["image_id", *attributes],
        nrows=limit, dtype={"image_id": str},
    )
    if len(labels) != limit or declared_count < limit:
        raise ValueError(f"CelebA source does not contain the requested first {limit:,} rows")
    labels[attributes] = (labels[attributes] > 0).astype("uint8")
    image_paths = [image_dir / image_id for image_id in labels["image_id"]]
    _require(image_paths)

    frame = pd.concat(
        [labels[attributes].reset_index(drop=True), _image_frame(image_paths, planar=True)], axis=1
    )
    destination = _output_path(output, "full_data.h5")
    _write_hdf(frame, destination, force)
    verify_dataset("faces", destination)
    return destination


def create_housing(source: Path, output: Path, force: bool) -> Path:
    import pandas as pd

    source = source.expanduser().resolve()
    csv_path, image_dir = source / "austinHousingData.csv", source / "homeImages"
    _require([csv_path, image_dir])
    frame = pd.read_csv(csv_path)
    image_paths = [image_dir / str(name) for name in frame["homeImage"]]
    _require(image_paths)
    frame = frame.drop(columns=[
        "zpid", "description", "streetAddress", "zipcode", "latitude", "longitude",
        "latest_saledate",
    ])
    for column in ("city", "homeType", "latestPriceSource"):
        frame[column] = frame[column].astype("category").cat.codes
    frame = frame.drop(columns=["homeImage"]).reset_index(drop=True)
    frame = pd.concat([frame, _image_frame(image_paths, planar=False)], axis=1)
    destination = _output_path(output, "housing_small.h5")
    _write_hdf(frame, destination, force)
    verify_dataset("housing", destination)
    return destination


def create_birds(source: Path, output: Path, force: bool) -> Path:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    source = source.expanduser().resolve()
    nested = source / "CUB_200_2011"
    if nested.is_dir():
        source = nested
    required = [source / "images.txt", source / "attributes.txt",
                source / "attributes" / "image_attribute_labels.txt", source / "images"]
    _require(required)
    images = {}
    for line in (source / "images.txt").read_text(encoding="utf-8").splitlines():
        image_id, relative = line.split(maxsplit=1)
        images[int(image_id)] = source / "images" / relative
    attributes = {}
    for line in (source / "attributes.txt").read_text(encoding="utf-8").splitlines():
        attribute_id, name = line.split(maxsplit=1)
        attributes[int(attribute_id)] = name
    labels = np.zeros((len(images), len(attributes)), dtype=np.uint8)
    for line in (source / "attributes" / "image_attribute_labels.txt").read_text(encoding="utf-8").splitlines():
        image_id, attribute_id, present, certainty, *_ = line.split()
        labels[int(image_id) - 1, int(attribute_id) - 1] = int(present) * (int(certainty) > 1)
    ordered_paths = [images[index] for index in sorted(images)]
    _require(ordered_paths)
    pixels = np.vstack([_image_vector(path, planar=True) for path in tqdm(ordered_paths, desc="CUB images")])
    frame = pd.concat([
        pd.DataFrame(labels, columns=[attributes[index] for index in sorted(attributes)]),
        pd.DataFrame(pixels, columns=_pixel_columns(planar=True)),
    ], axis=1)
    destination = _output_path(output, "full_data.h5")
    _write_hdf(frame, destination, force)
    verify_dataset("birds", destination)
    return destination


def _voc_root(source: Path) -> Path:
    source = source.expanduser().resolve()
    if (source / "JPEGImages").is_dir():
        return source
    matches = sorted(source.glob("VOCdevkit/VOC*/JPEGImages"))
    if matches:
        return matches[-1].parent
    raise FileNotFoundError("Expected JPEGImages/ and Annotations/ below the PASCAL VOC source")


def create_objects(source: Path, output: Path, force: bool) -> Path:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    source = _voc_root(source)
    _require([source / "JPEGImages", source / "Annotations"])
    image_paths = sorted(path for path in (source / "JPEGImages").iterdir()
                         if path.suffix.lower() in IMAGE_EXTENSIONS)
    labels, used_images = [], []
    for image_path in image_paths:
        annotation = source / "Annotations" / f"{image_path.stem}.xml"
        if not annotation.exists():
            continue
        classes = {node.text for node in ET.parse(annotation).findall(".//object/name")}
        labels.append([int(name in classes) for name in VOC_CLASSES])
        used_images.append(image_path)
    pixels = np.vstack([_image_vector(path, planar=True) for path in tqdm(used_images, desc="VOC images")])
    frame = pd.concat([
        pd.DataFrame(labels, columns=VOC_CLASSES, dtype="uint8"),
        pd.DataFrame(pixels, columns=_pixel_columns(planar=True)),
    ], axis=1)
    destination = _output_path(output, "full_data.h5")
    _write_hdf(frame, destination, force)
    verify_dataset("objects", destination)
    return destination


def create_music(source: Path, output: Path, force: bool) -> Path:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    source = source.expanduser().resolve()
    paths = {name: source / f"{name}.csv" for name in ("tracks", "genres", "features", "echonest")}
    _require(paths.values())
    tracks = pd.read_csv(paths["tracks"], index_col=0, header=[0, 1])
    for column in [("track", "tags"), ("album", "tags"), ("artist", "tags"),
                   ("track", "genres"), ("track", "genres_all")]:
        tracks[column] = tracks[column].map(ast.literal_eval)
    subsets = pd.CategoricalDtype(categories=("small", "medium", "large"), ordered=True)
    tracks[("set", "subset")] = tracks[("set", "subset")].astype(subsets)
    genres = pd.read_csv(paths["genres"], index_col=0)
    features = pd.read_csv(paths["features"], index_col=0, header=[0, 1, 2])
    echonest = pd.read_csv(paths["echonest"], index_col=0, header=[0, 1, 2])
    indices = echonest.index.intersection(tracks.index)
    useful_tracks = tracks.loc[indices, "track"][["duration", "genre_top"]].copy()
    useful_tracks["genre_top"] = useful_tracks["genre_top"].map(
        {title: index for index, title in genres["title"].items()}
    )
    useful_tracks["duration"] = pd.cut(
        useful_tracks["duration"], bins=list(range(0, 1001, 30)), labels=range(33)
    )
    echo = echonest.loc[indices, "echonest"][["audio_features", "social_features"]]
    discretized = pd.DataFrame(index=indices)
    for group, name in echo.columns:
        values = echo[(group, name)]
        if name == "tempo":
            discretized[name] = pd.cut(values, bins=list(range(0, 286, 25)), labels=range(11))
        else:
            discretized[name] = pd.cut(values, bins=[-1, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], labels=range(10))
    combined = pd.concat([useful_tracks, discretized, features.loc[indices]], axis=1).dropna()
    combined.columns = combined.columns.astype(str)
    numeric = list(combined.select_dtypes("float64").columns)
    categorical = sorted(column for column in combined.columns if column not in numeric)
    scaled = ColumnTransformer([("num", StandardScaler(), numeric)]).fit_transform(combined)
    frame = pd.DataFrame(
        np.concatenate((np.asarray(combined[categorical]), scaled), axis=1),
        index=combined.index, columns=categorical + numeric,
    )
    if "song_currency" in frame.columns:
        frame = frame.drop(columns="song_currency")
    destination = _output_path(output, "music.h5")
    _write_hdf(frame, destination, force)
    verify_dataset("music", destination, strict_columns=False)
    return destination


def _light_classification(data, windows: int, maximum: float, minimum: float, mean: float) -> list[int]:
    import numpy as np
    from scipy.signal import find_peaks

    result = [0] * (7 * windows)
    for window in range(windows):
        subsequence = data[window * math.ceil(len(data) / windows):(window + 1) * math.ceil(len(data) / windows)]
        local_mean = sum(subsequence) / len(subsequence)
        spread = max(subsequence) - min(subsequence)
        offset = 7 * window
        result[offset] = int(max(subsequence) - mean > .5 * (maximum - mean))
        result[offset + 1] = int(mean - min(subsequence) < .5 * (mean - minimum))
        close = sum(abs(value - local_mean) < .1 * spread for value in subsequence)
        result[offset + 2] = int(close > .5 * len(subsequence))
        result[offset + 3] = int(close < .1 * len(subsequence))
        spectrum = np.abs(np.fft.fft(subsequence)) ** 2
        positive = spectrum[:len(spectrum) // 2]
        result[offset + 4] = int(bool(len(find_peaks(positive, height=.1 * np.max(positive), distance=5)[0])))
        result[offset + 5] = int(local_mean > mean)
        result[offset + 6] = int(spread > .25 * abs(maximum - minimum))
    return result


def create_light(source: Path, output: Path, force: bool) -> Path:
    import pandas as pd
    from tqdm import tqdm

    source = source.expanduser().resolve()
    train, test = source / "StarLightCurves_TRAIN.txt", source / "StarLightCurves_TEST.txt"
    _require([train, test])
    rows = []
    for path in (train, test):
        with path.open(encoding="utf-8") as handle:
            rows.extend([[float(value) for value in line.split()][1:] for line in handle if line.strip()])
    if len({len(row) for row in rows}) != 1:
        raise ValueError("StarLightCurves rows do not have a consistent length")
    maximum, minimum = max(map(max, rows)), min(map(min, rows))
    mean = sum(map(sum, rows)) / sum(map(len, rows))
    windows = 3
    symbolic = [_light_classification(row, windows, maximum, minimum, mean)
                for row in tqdm(rows, desc="StarLightCurves")]
    names = [f"{name}_{window}" for window in range(windows) for name in LIGHT_FEATURES]
    frame = pd.concat([
        pd.DataFrame(symbolic, columns=names),
        pd.DataFrame(rows, columns=[f"t_{index}" for index in range(len(rows[0]))]),
    ], axis=1)
    destination = _output_path(output, "full_data.h5")
    _write_hdf(frame, destination, force)
    verify_dataset("light", destination)
    return destination


def create_motifs(output: Path, force: bool, seed: int = 0) -> Path:
    import numpy as np
    import pandas as pd
    from aeon.datasets import load_classification

    np.random.seed(seed)
    random.seed(seed)
    train_x, train_y = load_classification(name="NATOPS", split="train", load_equal_length=False)
    test_x, test_y = load_classification(name="NATOPS", split="test", load_equal_length=False)
    classes: dict[str, list[list[float]]] = {}
    for values, label in zip(list(train_x) + list(test_x), list(train_y) + list(test_y), strict=True):
        classes.setdefault(label, []).append([float(value[0]) for value in values.T])
    available_names = list("ABCDEFGHIJKLMNOPQRSTUVW")
    shifted = {}
    for label in classes:
        base = classes[label]
        for delta in (0, 2, -2):
            name = available_names.pop(0)
            shifted[name] = [[value + delta for value in row] for row in base]
    names = list(shifted)
    series_length = len(shifted[names[0]][0])
    columns = [f"{name}{segment}" for segment in range(10) for name in names]
    columns += [f"t{index}" for index in range(series_length * 10)]
    data = []
    for _ in range(5_000):
        logical, series = [], []
        for _ in range(10):
            chosen = random.randrange(len(names))
            logical.extend(int(index == chosen) for index in range(len(names)))
            series.extend(random.choice(shifted[names[chosen]]))
        data.append(logical + series)
    destination = _output_path(output, "full_data.h5")
    _write_hdf(pd.DataFrame(data, columns=columns), destination, force)
    verify_dataset("motifs", destination)
    return destination


def hdf_metadata(path: Path) -> tuple[int, list[str]]:
    """Read fixed-format pandas HDF metadata without loading the data matrix."""
    import tables

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with tables.open_file(path, mode="r") as handle:
        try:
            group = handle.root.full_data
            rows = int(group.axis1.shape[0])
            columns = [value.decode() if isinstance(value, bytes) else str(value)
                       for value in group.axis0.read()]
        except Exception as error:
            raise ValueError(f"{path} is not a pandas fixed-format HDF5 file with key 'full_data'") from error
    return rows, columns


def verify_dataset(dataset: str, path: Path, strict_columns: bool = True) -> None:
    expected = {
        "housing": (5_508, 39, 64), "music": (9_217, 14, None),
        "birds": (11_788, 312, 64), "objects": (17_125, 20, 64),
        "faces": (10_000, 40, 64), "medical": (1_011, 13, 64),
        "light": (9_236, 21, 1024), "motifs": (5_000, 180, 510),
        "medical-diagnosis": (1_011, 41, 512),
    }
    if dataset not in expected:
        raise ValueError(f"No verifier registered for {dataset}")
    rows, symbolic_count, subsymbolic = expected[dataset]
    actual_rows, columns = hdf_metadata(path)
    if actual_rows != rows:
        raise ValueError(f"{path}: expected {rows:,} rows, found {actual_rows:,}")
    if dataset == "music":
        if len(columns) != 532 or len(columns[:symbolic_count]) != symbolic_count:
            raise ValueError(f"{path}: unexpected FMA schema")
    elif dataset == "light":
        if columns[:21] != [f"{name}_{window}" for window in range(3) for name in LIGHT_FEATURES]:
            raise ValueError(f"{path}: unexpected StarLight symbolic columns")
        if len(columns) != symbolic_count + subsymbolic:
            raise ValueError(f"{path}: unexpected StarLight column count")
    elif dataset == "motifs":
        if len(columns) != symbolic_count + subsymbolic:
            raise ValueError(f"{path}: unexpected motif column count")
    elif dataset == "medical-diagnosis":
        expected_total = symbolic_count + 2 * 3 * subsymbolic * subsymbolic
        if len(columns) != expected_total:
            raise ValueError(f"{path}: expected {expected_total:,} columns, found {len(columns):,}")
    else:
        expected_total = symbolic_count + 3 * subsymbolic * subsymbolic
        if len(columns) != expected_total:
            raise ValueError(f"{path}: expected {expected_total:,} columns, found {len(columns):,}")
        if strict_columns and dataset == "faces" and columns[:40] != CELEBA_ATTRIBUTES:
            raise ValueError(f"{path}: unexpected CelebA attributes")
        if strict_columns and dataset == "objects" and set(columns[:20]) != set(VOC_CLASSES):
            raise ValueError(f"{path}: unexpected PASCAL VOC classes")
    print(f"Verified {dataset}: {actual_rows:,} rows x {len(columns):,} columns at {path}")


def copy_metadata_files(source: Path, output: Path, names: Iterable[str], force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        src, dst = source / name, output / name
        _require([src])
        if force or not dst.exists():
            shutil.copyfile(src, dst)
