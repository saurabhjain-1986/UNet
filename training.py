import os
import random

from models import build_compile_model
from training_utils import get_dataset_training, get_dataset_validation, my_callbacks


class Trainer:

    def __init__(self, path=None, pretrained_model=None):
        self.path = path
        self.input_dir = os.path.join(self.path, "images/")
        self.target_dir = os.path.join(self.path, "annotations/trimaps/")
        self.img_size = (128, 128)
        self.num_classes = 3
        self.batch_size = 200
        self.val_samples = 1000
        self.pretrained_model = pretrained_model
        self.model = build_compile_model()
        input_img_paths = sorted([os.path.join(self.input_dir, fname) for fname in os.listdir(self.input_dir)
                                  if fname.endswith(".jpg")])
        target_img_paths = sorted([os.path.join(self.target_dir, fname) for fname in os.listdir(self.target_dir)
                                   if fname.endswith(".png") and not fname.startswith(".")])

        random.Random(1337).shuffle(input_img_paths)
        random.Random(1337).shuffle(target_img_paths)
        self.train_input_img_paths = input_img_paths[:-self.val_samples]
        self.train_target_img_paths = target_img_paths[:-self.val_samples]
        self.val_input_img_paths = input_img_paths[-self.val_samples:]
        self.val_target_img_paths = target_img_paths[-self.val_samples:]
        self.train_dataset = get_dataset_training(self.batch_size,
                                                  self.train_input_img_paths, self.train_target_img_paths,
                                                  max_dataset_len=None)
        self.valid_dataset = get_dataset_validation(self.batch_size,
                                                    self.val_input_img_paths, self.val_target_img_paths,
                                                    max_dataset_len=None)

    def run_train(self):
        if self.pretrained_model:
            self.model.load_weights(self.pretrained_model, by_name=True)
        self.model.fit(self.train_dataset, epochs=50,
                       validation_data=self.valid_dataset,
                       callbacks=my_callbacks(self.path))


if __name__ == '__main__':
    obj = Trainer(path='/home/saurabh/Desktop/segmentation',
                  pretrained_model=None)
    obj.run_train()
