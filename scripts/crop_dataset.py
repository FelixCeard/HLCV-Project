import glob
import os
import re
from PIL import Image
import tqdm

from multiprocessing.dummy import Pool as ThreadPool

# hyper-parameter
# path_to_dataset_root = 'C:/Users/felix/PycharmProjects/HLCV-Project/dataset/'
path_to_dataset_root = 'C:/Users/felix/Pictures'
path_to_dataset_root = 'F:/you_know_what_project_this_is/scraping/alldata'
extentions = ['png', 'jpg', 'jpeg', 'gif']

global path_to_output
global output_size
path_to_output = 'C:/Users/felix/PycharmProjects/HLCV-Project/test_cropping'
output_size = (360, 240)

# get all the files



# prepare the paths
def prepare_path(path, folder=None):
    if os.path.isdir(path) == False:
        splited = re.split(r'/|\\', path)
        last_folder = splited[-1]
        prepare_path('/'.join(splited[:-1]), last_folder)

    if folder is not None and os.path.exists(os.path.join(path, folder)) == False:
        os.mkdir(os.path.join(path, folder))


prepare_path(path_to_output)

def resize_function(file_path):
    img = Image.open(file_path)
    img = img.resize(output_size)

    # get file name
    file_name = re.split(r'/|\\', file_path)[-1]
    if len(file_name.replace(' ', '')) == 0:
        file_name = re.split(r'/|\\', file_path)[-2]

    img.save(os.path.join(path_to_output, file_name))

if __name__ == '__main__':
    files = []
    for extention in extentions:
        files.extend(glob.glob(os.path.join(path_to_dataset_root, f'**/*.{extention}')))

    print('found', len(files), 'files')

    pool = ThreadPool(6)
    results = pool.map(resize_function, files)
    pool.close()
    pool.join()
    


# for file_path in tqdm.tqdm(files):
#     img = Image.open(file_path)
#     img = img.resize(output_size)
#
#     # get file name
#     file_name = re.split(r'/|\\', file_path)[-1]
#     if len(file_name.replace(' ', '')) == 0:
#         file_name = re.split(r'/|\\', file_path)[-2]
#
#     img.save(os.path.join(path_to_output, file_name))

