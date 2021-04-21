import requests
from tqdm import tqdm
import zipfile
import os
import shutil


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
