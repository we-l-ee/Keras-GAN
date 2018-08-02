


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
    X, y = [], []
    for path in paths:
        im = cv2.resize(cv2.imread(path), shape)
        # print(im.shape)
        X.append(im)
        y.append(encoder(path))

    return np.array(X), np.array(y)


def encoder(file):
    return 0 if splitext(basename(file))[0][-2:] == 'OK' else 1

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


def train(config, gan):
    train_on = {"OK": get_files_OK, "FAULT": get_files_FAULT, "all": get_files}

    files = train_on[config.get("General", "train_on")](config)

    data = load_imagefiles(files,
                        (config.getint("Model","rows"), config.getint("Model","cols")))

    gan.train(data, epochs=config.getint("Train","epochs"), batch_size=config.getint("Train","batch_size"),
              sample_interval=config.getint("Train","sample_interval"))
import datetime

def imwrite(path, im):
    im = im*255
    im = im.astype(np.uint8)
    cv2.imwrite(path, im)

def generate(config, gan):
    feed_ones = ["DUALGAN", 'Pix2Pix']
    if config.get("General", "model") in feed_ones:
        gan.feed(config.get("General", "feed"))
    imgs = gan.generate(config.getint("generate","nums"))
    output_folder = config.get('generate','output_folder')
    for i, img in enumerate(imgs):
        imwrite(join(output_folder,str(datetime.datetime.now())+str(i)),img)

def main():
    config = configparser.ConfigParser()
    config.read("runner.cfg")
    models = {"DUALGAN": DUALGAN, "WGAN":WGAN, "BIGAN":BIGAN, "Pix2Pix":Pix2Pix}
    modes = {'train':train, 'generate':generate}

    #
    # ok, fault = splitByClass(getImageFiles(config.get("Model", "input_folder"), recursive=True))
    # print(ok)
    # print(fault)

    gan = models[config.get("General", "model")](config=config)


    modes[config.get('General','mode')](config, gan)


if __name__ == '__main__':
    main()


