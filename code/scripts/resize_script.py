import pandas as pd
from keras.preprocessing.image import load_img
from PIL import ImageFile

# ask PIL to be tolerant of files that are truncated
# ie, if image has missing data, it will be filled with gray
# suboptimal - would like to know which images are affected... also should've timed this
# https://stackoverflow.com/a/23575424
ImageFile.LOAD_TRUNCATED_IMAGES = True

# path to csv
csv_path = '../painting_data/kaggle_paint_numbers/csv/all_data_info.csv'
all_data = pd.read_csv(csv_path)

# subset only paintings in training set
train_paintings = all_data[all_data.in_train]


# file indices, names
num_files = range(train_paintings.shape[0])
filelist = train_paintings.index.values
filenames = train_paintings.new_filename

# source and destination directories
source_path = '../painting_data/kaggle_paint_numbers/paintings_train/'
dest_path = '../painting_data/kaggle_paint_numbers/resized_72/'

# loop over images and resize
for n, i, x in zip(num_files, filelist, filenames):
    train_img = load_img(source_path + filenames[i])
    train_img_resized = train_img.resize((72, 72))
    train_img_resized.save(dest_path + filenames[i])
    if n % 1000 == 0:
        print(n)
