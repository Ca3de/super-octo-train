import os
import random
import scipy.io
import numpy as np
import cv2 as cv
from console_progressbar import ProgressBar
from .data_utils import ensure_folder

IMG_WIDTH, IMG_HEIGHT = 224, 224

def save_train_data(fnames, labels, bboxes, src_folder, train_ratio=0.8, margin=16):
    """Save training and validation data with bounding boxes."""
    num_samples = len(fnames)
    num_train = int(round(num_samples * train_ratio))
    train_indexes = random.sample(range(num_samples), num_train)
    
    pb = ProgressBar(total=100, prefix='Save Train Data', suffix='', decimals=3, length=50, fill='=')
    for i, fname in enumerate(fnames):
        label, (x1, y1, x2, y2) = labels[i], bboxes[i]
        src_path = os.path.join(src_folder, fname)
        image = cv.imread(src_path)
        
        # Add margins and crop
        height, width = image.shape[:2]
        x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(x2 + margin, width), min(y2 + margin, height)
        cropped = image[y1:y2, x1:x2]
        resized = cv.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
        
        # Save to train or validation
        dst_folder = '/kaggle/working/data/train' if i in train_indexes else '/kaggle/working/data/valid'
        dst_path = os.path.join(dst_folder, label, fname)
        ensure_folder(os.path.dirname(dst_path))
        cv.imwrite(dst_path, resized)
        pb.print_progress_bar((i + 1) * 100 / num_samples)

# Save test data function
def save_test_data(fnames, bboxes, src_folder):
    """Save test data with bounding boxes."""
    dst_folder = '/kaggle/working/data/test'
    num_samples = len(fnames)
    
    pb = ProgressBar(total=100, prefix='Save Test Data', suffix='', decimals=3, length=50, fill='=')
    for i, fname in enumerate(fnames):
        x1, y1, x2, y2 = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        image = cv.imread(src_path)
        
        # Add margins and crop
        height, width = image.shape[:2]
        x1, y1, x2, y2 = max(0, x1 - 16), max(0, y1 - 16), min(x2 + 16, width), min(y2 + 16, height)
        cropped = image[y1:y2, x1:x2]
        resized = cv.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
        
        dst_path = os.path.join(dst_folder, fname)
        ensure_folder(dst_folder)
        cv.imwrite(dst_path, resized)
        pb.print_progress_bar((i + 1) * 100 / num_samples)

# Process annotation function
def process_annotations(annotation_path, src_folder, is_test=False):
    """Process annotation data from .mat files."""
    annotations = scipy.io.loadmat(annotation_path)['annotations']
    annotations = np.transpose(annotations)
    
    fnames, labels, bboxes = [], [], []
    for annotation in annotations:
        x1, y1, x2, y2 = annotation[0][0][0][0], annotation[0][1][0][0], annotation[0][2][0][0], annotation[0][3][0][0]
        fname = annotation[0][4][0] if is_test else annotation[0][5][0]
        label = '%04d' % annotation[0][4][0] if not is_test else None
        bboxes.append((x1, y1, x2, y2))
        fnames.append(fname)
        if not is_test:
            labels.append(label)
    return fnames, labels, bboxes
