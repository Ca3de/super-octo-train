from fastai.vision import *
from fastai.metrics import accuracy
from model.focal_loss import FocalLoss
from preprocess.preprocess_data import process_annotations, save_train_data

# Paths for the updated dataset structure
TRAIN_ANNOTATIONS_PATH = '/kaggle/input/devkit/cars_train_annos.mat'
TRAIN_IMAGES_PATH = '/kaggle/input/stanford-cars-dataset/cars_train'
PRETRAINED_MODEL_PATH = '/kaggle/input/se-resnext50-32x4d-a260b3a4-pth/se_resnext50_32x4d-a260b3a4.pth'

def prepare_data():
    """Prepare data by processing annotations and saving cropped images."""
    fnames, labels, bboxes = process_annotations(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGES_PATH)
    save_train_data(fnames, labels, bboxes, TRAIN_IMAGES_PATH)

def train_model():
    """Train a model using fastai."""
    data = ImageDataBunch.from_folder('/kaggle/working/data/', train='train', valid='valid', size=224, bs=64).normalize(imagenet_stats)
    learn = cnn_learner(data, models.resnet50, metrics=[accuracy])
    learn.loss_func = FocalLoss()
    learn.fit_one_cycle(10, max_lr=1e-3)
    learn.save('model-stage-1')

if __name__ == '__main__':
    prepare_data()
    train_model()
