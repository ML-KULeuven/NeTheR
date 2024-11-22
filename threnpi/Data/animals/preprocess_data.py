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
    animal_data = pd.DataFrame(columns=["animal"] + predicates)
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