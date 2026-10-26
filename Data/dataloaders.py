import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import ast

DATA_DIR = Path(__file__).resolve().parent



def load_animal_data():
    #AWA2dataset
    categorical_columns = []
    with (DATA_DIR / "animals" / "predicates.txt").open(encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.split("\t")
        categorical_columns.append(splitted[1][:-1])
    droppable_cols = categorical_columns.copy()
    sub = []
    for i in range(0, 8):
        sub.append(pd.read_hdf(DATA_DIR / "animals" / f"full_data_{i}_64.h5", key='full_data', mode='r'))
    intended = ["animal"] + categorical_columns.copy()
    for i in ["r", "g", "b"]:
        for x in range(0, 64):
            for y in range(0, 64):
                intended.append(i + "_" + str(x) + "_" + str(y))
    full_data = pd.concat(sub, axis=0).reset_index(drop=True)[intended[1:]]
    return full_data, categorical_columns, droppable_cols


def animal_image(data, index):
    """Return one processed AWA2 image as a 64 x 64 RGB array.

    ``index`` is a dataframe label, matching the saved problem split indices.
    """
    pixels = data.loc[index].iloc[-3 * 64 * 64:]
    return np.asarray(pixels, dtype=np.uint8).reshape(3, 64, 64).transpose(1, 2, 0)


def show_animal(data, index, ax=None):
    """Display one processed AWA2 image and return its Matplotlib axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(animal_image(data, index))
    ax.axis("off")
    return ax


# show_animal(small_data,1369)
def load_small_animal_data(return_animal_names=False):
    categorical_columns = []
    with (DATA_DIR / "animals" / "predicates.txt").open(encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.split("\t")
        categorical_columns.append(splitted[1][:-1])
    droppable_cols = categorical_columns.copy()
    sub = []
    for i in range(0, 8):
        sub.append(pd.read_hdf(DATA_DIR / "animals" / f"full_data_{i}_64.h5", key='full_data', mode='r'))
    intended = ["animal"] + categorical_columns.copy()
    for i in ["r", "g", "b"]:
        for x in range(0, 64):
            for y in range(0, 64):
                intended.append(i + "_" + str(x) + "_" + str(y))
    full_data = pd.concat(sub, axis=0).reset_index(drop=True)[intended]
    classes = []
    with (DATA_DIR / "animals" / "classes.txt").open(encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.split("\t")
        classes.append(splitted[1][:-1])
    small_index = []
    for an in classes:
        small_index += list(full_data.loc[full_data['animal'] == an].index[:100])
    selected = full_data.iloc[small_index].reset_index(drop=True)
    animal_names = selected["animal"].copy()
    small_data = selected.drop(columns=["animal"])
    if return_animal_names:
        return small_data, categorical_columns, droppable_cols, animal_names
    return small_data, categorical_columns, droppable_cols

import pandas as pd

def load_housing_data():
    #austin texas housing dataset (kaggle)
    droppable_cols = ['hasSpa', 'hasView',
                      'homeType', 'parkingSpaces', 'yearBuilt', 'latestPrice', 'numOfParkingFeatures',
                      'numOfPatioAndPorchFeatures',
                      'numOfSecurityFeatures', 'numOfWaterfrontFeatures',
                      'numOfWindowFeatures', 'lotSizeSqFt',
                      'livingAreaSqFt', 'numOfBathrooms', 'numOfBedrooms', 'numOfStories']
    cat_cols = ['city', 'propertyTaxRate', 'hasAssociation',
                'hasCooling', 'hasHeating', 'hasSpa', 'hasView',
                'homeType', 'parkingSpaces', 'yearBuilt', 'latestPrice',
                'numPriceChanges', 'latest_salemonth', 'latest_saleyear',
                'latestPriceSource', 'numOfPhotos', 'numOfAccessibilityFeatures',
                'numOfAppliances', 'numOfParkingFeatures', 'numOfPatioAndPorchFeatures',
                'numOfSecurityFeatures', 'numOfWaterfrontFeatures',
                'numOfWindowFeatures', 'numOfCommunityFeatures', 'lotSizeSqFt',
                'livingAreaSqFt', 'numOfPrimarySchools', 'numOfElementarySchools',
                'numOfMiddleSchools', 'numOfHighSchools', 'avgSchoolDistance',
                'avgSchoolRating', 'avgSchoolSize', 'MedianStudentsPerTeacher',
                'numOfBathrooms', 'numOfBedrooms', 'numOfStories']
    full_data=pd.read_hdf(DATA_DIR / "housing" / "housing_small.h5",key='full_data',mode='r')
    full_data.drop(columns=["garageSpaces", "hasGarage"], inplace=True)
    sorted_colours=sorted(list(set(full_data.columns)-set(cat_cols)), key=lambda x: (-ord(x[0]), int(x[2:x.find("_",2)]),int(x[x.find("_",2)+1:])))
    full_data=full_data[list(cat_cols)+sorted_colours].reset_index(drop=True)

    return full_data, cat_cols, droppable_cols

def load_music_data():
    #FMA dataset
    # Load metadata and features.
    full_data=pd.read_hdf(DATA_DIR / "music" / "music.h5", key="full_data", mode="r").reset_index(drop=True)

    categorical_columns = list(full_data.columns)[:14]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_birds_data():
    #CUB dataset
    # Load metadata and features.
    full_data = pd.read_hdf(DATA_DIR / "birds" / "full_data.h5", key="full_data", mode="r").reset_index(drop=True)

    categorical_columns = list(full_data.columns)[:312]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols



def load_objects_data():
    #VOC PASCAL dataset
    # Load metadata and features.
    full_data = pd.read_hdf(DATA_DIR / "objects" / "full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:20]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_faces_data():
    #CELEBA dataset
    # Load metadata and features.
    full_data = pd.read_hdf(DATA_DIR / "faces" / "full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    full_data = full_data.rename(columns={'5_o_Clock_Shadow': 'five_o_Clock_Shadow'})
    categorical_columns = list(full_data.columns)[:40]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_medical_data():
    #CELEBA dataset
    # Load metadata and features.
    full_data = pd.read_hdf(DATA_DIR / "medical" / "full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:13]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_medical_diagnosis_data():
    # Load metadata and features.
    full_data = pd.read_hdf(DATA_DIR / "medical" / "full_data_diagnosis.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:41]
    categorical_columns.remove("diagnosis")
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_medical_diagnosis_meta_data():
    # Load metadata and features.
    full_data = pd.read_hdf(DATA_DIR / "medical" / "full_data_diagnosis.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = ['elevation_0', 'elevation_1', 'elevation_2', 'level_of_diagnostic_difficulty_0', 'level_of_diagnostic_difficulty_1', 'level_of_diagnostic_difficulty_2', 'location_0', 'location_1', 'location_2', 'location_3', 'location_4', 'location_5', 'location_6', 'location_7', 'location_8', 'management_0', 'management_1', 'management_2','sex_0', 'sex_1']
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols


def load_medical_diagnosis_concepts_data():
    #CELEBA dataset
    # Load metadata and features.
    full_data = pd.read_hdf(DATA_DIR / "medical" / "full_data_diagnosis_predicted_concepts.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:-6*512*512]

    categorical_columns.remove("diagnosis")
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols




def load_light_data():
    full_data = pd.read_hdf(DATA_DIR / "light" / "full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:21]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_motif_data():
    full_data = pd.read_hdf(DATA_DIR / "motifs" / "full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:180]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols


def load_data(data):
    if data=="animals":
        return load_small_animal_data()
    elif data=="housing":
        return load_housing_data()
    elif data=="music":
        return load_music_data()
    elif data=="birds":
        return load_birds_data()
    elif data=="faces":
        return load_faces_data()
    elif data=="objects":
        return load_objects_data()
    elif data=="medical":
        return load_medical_data()
    elif data=="medical_diagnosis":
        return load_medical_diagnosis_data()
    elif data=="medical_concepts":
        return load_medical_diagnosis_concepts_data()
    elif data == "medical_meta":
        return load_medical_diagnosis_meta_data()
    elif data=="light":
        return load_light_data()
    elif data=="motifs":
        return load_motif_data()
