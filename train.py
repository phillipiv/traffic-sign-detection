import os

from lib import train_detector


DATA_DIR = os.path.join('.', 'data')

img_dir = os.path.join(DATA_DIR, 'train', 'imgs')
label_dir = os.path.join(DATA_DIR, 'train', 'anns')

img_dir_val = os.path.join(DATA_DIR, 'validation', 'imgs')
label_dir_val = os.path.join(DATA_DIR, 'validation', 'anns')

class_list_file = 'classes.txt'

gtf = train_detector.Detector()

gtf.set_train_dataset(img_dir, label_dir, class_list_file, batch_size=32)
gtf.set_val_dataset(img_dir_val, label_dir_val)

gtf.set_model(model_name="yolov3")

gtf.set_hyperparams(optimizer="sgd", lr=0.00579);

gtf.Train(num_epochs=20)
