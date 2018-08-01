


import configparser
import cv2
import glob
import numpy as np
from os.path import join, splitext, basename
from dualgan.dualgan import DUALGAN
from wgan.wgan import WGAN
from bigan.bigan import BIGAN
from pix2pix.pix2pix import Pix2Pix

def getImageFiles(input_folder, exts = (".jpg", ".gif", ".png", ".tga", ".tif"), recursive=False):
    files = []
    for ext in exts:
        files.extend(glob.glob(join(input_folder,'*%s' % ext), recursive=recursive))
    return files

def load_imagefiles(paths, shape=(100,100)):
    data = []
    for path in paths:
        im = cv2.resize(cv2.imread(path), shape)
        # print(im.shape)
        data.append(im)

    return np.array(data)


def splitByClass(files):
    fault, ok = [], []
    for file in files:
        if splitext(basename(file))[0][-2:] == 'OK':
            ok.append(file)
        else:
            fault.append(file)
    return ok, fault

def get_files_OK(config):
    ok, _ = splitByClass(getImageFiles(config.get("Model", "input_folder"), recursive=True))
    return ok

def get_files_FAULT(config):
    _, fault = splitByClass(getImageFiles(config.get("Model", "input_folder"), recursive=True))
    return fault

def get_files(config):
    return getImageFiles(config.get("Model","input_folder"), recursive=True)


def main():
    config = configparser.ConfigParser()
    config.read("runner.cfg")
    models = {"DUALGAN": DUALGAN, "WGAN":WGAN, "BIGAN":BIGAN, "Pix2Pix":Pix2Pix}

    train_on = {"OK": get_files_OK, "FAULT": get_files_FAULT, "all": get_files}

    #
    # ok, fault = splitByClass(getImageFiles(config.get("Model", "input_folder"), recursive=True))
    # print(ok)
    # print(fault)

    gan = models[config.get("General", "model")](config=config)

    files = train_on[config.get("General", "train_on")](config)

    X = load_imagefiles(files,
                        (config.getint("Model","rows"), config.getint("Model","cols")))

    gan.train(X, epochs=config.getint("Train","epochs"), batch_size=config.getint("Train","batch_size"),
              sample_interval=config.getint("Train","sample_interval"))

if __name__ == '__main__':
    main()


