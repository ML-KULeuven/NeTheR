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
#------------------ ANIMALS ----------------
# categorical_columns = []
# with open("data/animals/predicates.txt") as f:
#     lines = f.readlines()
# for line in lines:
#     splitted = line.split("\t")
#     categorical_columns.append(splitted[1][:-1])
# droppable_cols = categorical_columns.copy()
# classes = []
# with open("data/animals/classes.txt") as f:
#     lines = f.readlines()
# for line in lines:
#     splitted = line.split("\t")
#     classes.append(splitted[1][:-1])
# with open("data/animals/predicate-matrix-binary.txt") as f:
#     lines = f.readlines()
# all_preds = []
# for line in lines:
#     all_preds.append([int(x) for x in line[:-1].split(" ")])
#
# for it in range(0, 8):
#     new_df = [[] for x in range(64 * 64 * 3)]
#     class_list = []
#     for img in tqdm(list_all_files("data/animals/JPEGImages/")[it * 5000:(it + 1) * 5000]):
#         class_list.append(img[img.find("\\") + 1:img.find("_")])
#         image = process_image(img)
#         im_array = np.asarray(image.convert('RGB'))
#         counter = 0
#         for i in range(64):
#             for x in range(64):
#                 for c in range(3):
#                     new_df[counter].append(im_array[i][x][c])
#                     counter += 1
#     animal_data = pd.DataFrame(columns=["animal"] + predicates)
#     for animal in tqdm(class_list):
#         animal_id = classes.index(animal) - 1
#         animal_data.loc[len(animal_data.index)] = [animal] + all_preds[animal_id]
#     counter = 0
#     simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
#     for i in tqdm(range(64)):
#         for x in range(64):
#             for c in range(3):
#                 animal_data[["r", "g", "b"][c] + "_" + str(i) + "_" + str(x)] = new_df[counter]
#                 counter += 1
#     animal_data.to_hdf("data/animals/full_data_" + str(it) + "_64.h5", key='full_data', mode='w', )
#
#
#----------------------HOUSING-------------------------
# from warnings import simplefilter
#housing_data=pd.read_csv("data/housing/austinHousingData.csv")
# housing_data=housing_data.drop(columns=["zpid","description","streetAddress","zipcode","latitude","longitude","latest_saledate"])
# housing_data.city=housing_data.city.astype('category').cat.codes
# housing_data.homeType=housing_data.homeType.astype('category').cat.codes
# housing_data.latestPriceSource=housing_data.latestPriceSource.astype('category').cat.codes
# categorical_columns=housing_data.drop(columns=["homeImage"]).columns
# droppable_cols=['hasCooling', 'hasGarage', 'hasHeating', 'hasSpa', 'hasView',
#        'homeType', 'parkingSpaces', 'yearBuilt', 'latestPrice','numOfParkingFeatures', 'numOfPatioAndPorchFeatures',
#        'numOfSecurityFeatures', 'numOfWaterfrontFeatures',
#        'numOfWindowFeatures','lotSizeSqFt',
#        'livingAreaSqFt','numOfBathrooms', 'numOfBedrooms', 'numOfStories']
# housing_data=housing_data.query("garageSpaces==2") #go back to ==2
# new_df = [[] for x in range(64 * 64 * 3)]
#
# for img in tqdm(list(housing_data["homeImage"])):
#     image = process_image("data/housing/homeImages/" + img)
#     im_array = np.asarray(image.convert('RGB'))
#     counter = 0
#     for i in range(64):
#         for x in range(64):
#             for c in range(3):
#                 new_df[counter].append(im_array[i][x][c])
#                 counter += 1
# housing_data = housing_data.drop(columns=["homeImage"])
# counter = 0
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# for i in tqdm(range(64)):
#     for x in range(64):
#         for c in range(3):
#             housing_data[["r", "g", "b"][c] + "_" + str(i) + "_" + str(x)] = new_df[counter]
#             counter += 1
# housing_data.to_hdf("data/housing/housing_small.h5", key='full_data', mode='w', )


#------------------------------MUSIC------------------------------
# tracks = pd.read_csv('Data/music/tracks.csv', index_col=0, header=[0, 1])
#     COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
#                ('track', 'genres'), ('track', 'genres_all')]
#     for column in COLUMNS:
#         tracks[column] = tracks[column].map(ast.literal_eval)
#
#     COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
#                ('album', 'date_created'), ('album', 'date_released'),
#                ('artist', 'date_created'), ('artist', 'active_year_begin'),
#                ('artist', 'active_year_end')]
#     for column in COLUMNS:
#         tracks[column] = pd.to_datetime(tracks[column])
#
#     SUBSETS = ('small', 'medium', 'large')
#     try:
#         tracks['set', 'subset'] = tracks['set', 'subset'].astype(
#             'category', categories=SUBSETS, ordered=True)
#     except (ValueError, TypeError):
#         # the categories and ordered arguments were removed in pandas 0.25
#         tracks['set', 'subset'] = tracks['set', 'subset'].astype(
#             pd.CategoricalDtype(categories=SUBSETS, ordered=True))
#
#     COLUMNS = [('track', 'genre_top'), ('track', 'license'),
#                ('album', 'type'), ('album', 'information'),
#                ('artist', 'bio')]
#     for column in COLUMNS:
#         tracks[column] = tracks[column].astype('category')
#     genres = pd.read_csv('Data/music/genres.csv', index_col=0)
#     features = pd.read_csv('Data/music/features.csv', index_col=0, header=[0, 1, 2])
#     echonest = pd.read_csv('Data/music/echonest.csv', index_col=0, header=[0, 1, 2])
#
#     np.testing.assert_array_equal(features.index, tracks.index)
#     assert echonest.index.isin(tracks.index).all()
#
#     tracks.shape, genres.shape, features.shape, echonest.shape
#     large = tracks[tracks['set', 'subset'] <= 'large']
#     small_tracks = large['track'][['duration', 'genre_top']]
#     small_indices = list(small_tracks.index)
#
#     echonest_indices = echonest.index
#     useful_indices = echonest_indices.intersection(small_indices)
#     useful_tracks = small_tracks.loc[useful_indices]
#
#     # Map each genre to its ID from the genre table
#     useful_tracks["genre_top"] = useful_tracks["genre_top"].transform(
#         lambda x: genres.index[genres['title'] == x].tolist()[0])
#
#     # Discretisize the duration per 30s
#     useful_tracks["duration"] = pd.cut(useful_tracks["duration"],
#                                        bins=list(range(0, 1001, 30)),
#                                        labels=list(range(0, len(range(0, 1001, 30)) - 1)))
#     useful_audio_features = features.loc[useful_indices]
#     echonest_features = echonest.loc[useful_indices]
#     useful_echonest_features = echonest_features['echonest'][['audio_features', 'social_features']]
#     discretized_echonest = pd.DataFrame()
#     for i in useful_echonest_features.columns:
#         # discretisize tempo per 25bpm
#         if i[1] == 'tempo':
#             discretized_echonest[i[1]] = pd.cut(useful_echonest_features[i[0]][i[1]],
#                                                 bins=list(range(0, 286, 25)),
#                                                 labels=list(range(0, len(range(0, 286, 25)) - 1)))
#         else:
#             # discretize all echonest features per 0.1
#             discretized_echonest[i[1]] = pd.cut(useful_echonest_features[i[0]][i[1]],
#                                                 bins=[-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#                                                 labels=list([x for x in range(0, 10)]))
#     # Add track_data, echonest features and audio features together
#     full_data = pd.concat([useful_tracks, discretized_echonest, useful_audio_features], axis=1).dropna()
#     full_data.columns = full_data.columns.astype(str)
#     # Normalize the continous data (discretization of echonest and track data happens above)
#     numeric_columns = list(full_data.select_dtypes('float64').columns)
#     categorical_columns = list(set(full_data.columns) - set(numeric_columns))
#     pipeline = ColumnTransformer([('num', StandardScaler(), numeric_columns)])
#
#     new_X = np.concatenate((np.array(full_data[categorical_columns]), pipeline.fit_transform(full_data)), axis=1)
#     full_data = pd.DataFrame(data=new_X, index=full_data.index, columns=categorical_columns + numeric_columns).drop(
#         columns="song_currency")
#     categorical_columns = list(set(full_data.columns) - set(numeric_columns))
#     droppable_cols = categorical_columns


#------------------BIRDS ----------------------------"
# id2img = {}
# img2id = {}
# with open("data/birds/images.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         splitted = line.split(" ")
#         id = int(splitted[0])
#         img = splitted[1].strip()
#         id2img[id] = img
#         img2id[img] = id
# classes = {}
# with open("data/birds/classes.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         splitted = line.split(" ")
#         id = int(splitted[0])
#         descr = splitted[1].strip()
#         classes[id] = descr
# id2class = {}
# with open("data/birds/image_class_labels.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         splitted = line.split(" ")
#         id = int(splitted[0])
#         class_id = int(splitted[1])
#         id2class[id] = class_id
# attributes = {}
# with open("data/birds/attributes.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         splitted = line.split(" ")
#         id = int(splitted[0])
#         descr = splitted[1].strip()
#         attributes[id] = descr
#
# all_dicts = []
# with open("data/birds/attributes/image_attribute_labels.txt") as f:
#     lines = f.readlines()
#     new_row = {}
#     for line in lines:
#         splitted = line.split(" ")
#         id = int(splitted[0])
#         att_id = int(splitted[1])
#         truth = int(splitted[2])
#         certainty = int(splitted[3])
#         new_row[attributes[att_id]] = truth * (certainty > 1)
#         if len(new_row) == 312:
#             all_dicts.append(new_row)
#             new_row = {}
#
# full_data = pd.DataFrame.from_records(all_dicts)
# new_df = [[] for x in range(64 * 64 * 3)]
# todo = [(id, id2img[id], classes[id2class[id]]) for id in id2img]
# for item in tqdm(todo):
#     img = "data/birds/images/" + item[1]
#     image = process_image(img)
#     im_array = np.asarray(image.convert('RGB'))
#     counter = 0
#     for i in range(64):
#         for x in range(64):
#             for c in range(3):
#                 new_df[counter].append(im_array[i][x][c])
#                 counter += 1
# counter = 0
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# for i in tqdm(range(64)):
#     for x in range(64):
#         for c in range(3):
#             full_data[["r", "g", "b"][c] + "_" + str(i) + "_" + str(x)] = new_df[counter]
#             counter += 1
# cat_cols = list(attributes.values())
# sorted_colours = sorted(list(set(full_data.columns) - set(cat_cols)),
#                         key=lambda x: (-ord(x[0]), int(x[2:x.find("_", 2)]), int(x[x.find("_", 2) + 1:])))
# final_data = full_data[list(cat_cols) + sorted_colours].reset_index(drop=True)
# final_data.to_hdf("data/birds/full_data.h5", key='full_data', mode='w', )

#---------------OBJECTS -----------------
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