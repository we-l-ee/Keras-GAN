


import configparser
import cv2
import glob
import numpy as np
from os.path import join
from dualgan.dualgan import DUALGAN

def getImageFiles(input_folder, exts = ("jpg", "gif", "png", "tga", "tif"), recursive=False):
    files = []
    for ext in exts:
        files.extend(glob.glob(join(input_folder,'*.%s' % ext), recursive=recursive))
    return files

def load_imagefiles(input_root, shape=(100,100)):
    data = []
    paths = getImageFiles(input_root, recursive=True)
    for path in paths:
        im = cv2.resize(cv2.imread(path), shape)
        # print(im.shape)
        data.append(im)

    return np.array(data)


def main():
    config = configparser.ConfigParser()
    config.read("runner.cfg")
    models = {"DUALGAN": DUALGAN}

    gan = models[config.get("General", "model")](config=config)
    X = load_imagefiles(config.get("Model","input_folder"),
                        (config.getint("Model","rows"), config.getint("Model","cols")))

    gan.train(X, epochs=config.getint("Train","epochs"), batch_size=config.getint("Train","batch_size"),
              sample_interval=config.getint("Train","sample_interval"))

if __name__ == '__main__':
    main()


