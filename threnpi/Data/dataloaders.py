import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import ast
def load_animal_data():
    #AWA2dataset
    categorical_columns = []
    with open("Data/animals/predicates.txt") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.split("\t")
        categorical_columns.append(splitted[1][:-1])
    droppable_cols = categorical_columns.copy()
    sub = []
    for i in range(0, 8):
        sub.append(pd.read_hdf("Data/animals/full_data_" + str(i) + "_64.h5", key='full_data', mode='r'))
    intended = ["animal"] + categorical_columns.copy()
    for i in ["r", "g", "b"]:
        for x in range(0, 64):
            for y in range(0, 64):
                intended.append(i + "_" + str(x) + "_" + str(y))
    full_data = pd.concat(sub, axis=0).reset_index(drop=True)[intended[1:]]
    return full_data, categorical_columns, droppable_cols


def show_animal(data, index):
    row = data.iloc[index][-4096 * 3:]
    # Stack the channels to create the image
    red_channel = row[-3 * 2048 * 2:-2 * 2048 * 2]
    blue_channel = row[-2 * 2048 * 2:-2048 * 2]
    green_channel = row[-2048 * 2:]
    test = np.array(row).reshape(1, 3, 64, 64)[0]

    # Stack the channels to create the image
    image = np.stack([red_channel, blue_channel, green_channel], axis=1).reshape(64, 64, 3)
    # plt.imshow(test[2].astype(np.uint8))
    # Display the image
    plt.imshow(image.astype(np.uint8))  # Ensure the pixel values are in uint8 format
    plt.axis('off')  # Hide the axis
    plt.show()


# show_animal(small_data,1369)
def load_small_animal_data():
    categorical_columns = []
    with open("Data/animals/predicates.txt") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.split("\t")
        categorical_columns.append(splitted[1][:-1])
    droppable_cols = categorical_columns.copy()
    sub = []
    for i in range(0, 8):
        sub.append(pd.read_hdf("Data/animals/full_data_" + str(i) + "_64.h5", key='full_data', mode='r'))
    intended = ["animal"] + categorical_columns.copy()
    for i in ["r", "g", "b"]:
        for x in range(0, 64):
            for y in range(0, 64):
                intended.append(i + "_" + str(x) + "_" + str(y))
    full_data = pd.concat(sub, axis=0).reset_index(drop=True)[intended]
    classes = []
    with open("Data/animals/classes.txt") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.split("\t")
        classes.append(splitted[1][:-1])
    small_index = []
    for an in classes:
        small_index += list(full_data.loc[full_data['animal'] == an].index[:100])
    small_data = full_data.iloc[small_index].reset_index(drop=True).drop(columns=["animal"])
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
    full_data=pd.read_hdf("Data/housing/housing_small.h5",key='full_data',mode='r')
    full_data.drop(columns=["garageSpaces", "hasGarage"], inplace=True)
    sorted_colours=sorted(list(set(full_data.columns)-set(cat_cols)), key=lambda x: (-ord(x[0]), int(x[2:x.find("_",2)]),int(x[x.find("_",2)+1:])))
    full_data=full_data[list(cat_cols)+sorted_colours].reset_index(drop=True)

    return full_data, cat_cols, droppable_cols

def load_music_data():
    #FMA dataset
    # Load metadata and features.
    full_data=pd.read_hdf("Data/music/music.h5", key="full_data", mode="r").reset_index(drop=True)

    categorical_columns = list(full_data.columns)[:14]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_birds_data():
    #CUB dataset
    # Load metadata and features.
    full_data = pd.read_hdf("Data/birds/full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:312]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_objects_data():
    #VOC PASCAL dataset
    # Load metadata and features.
    full_data = pd.read_hdf("Data/objects/full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:20]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_faces_data():
    #CELEBA dataset
    # Load metadata and features.
    full_data = pd.read_hdf("Data/faces/full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    full_data = full_data.rename(columns={'5_o_Clock_Shadow': 'five_o_Clock_Shadow'})
    categorical_columns = list(full_data.columns)[:40]
    droppable_cols = categorical_columns
    return full_data, categorical_columns, droppable_cols

def load_medical_data():
    #CELEBA dataset
    # Load metadata and features.
    full_data = pd.read_hdf("Data/medical/full_data.h5", key="full_data", mode="r").reset_index(drop=True)
    categorical_columns = list(full_data.columns)[:13]
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