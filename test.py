import os
import cv2
import random
import numpy as np

from result_utils import save_fig
from models import build_compile_model
from training_utils import get_dataset_validation, resample_image, largest_connected_component_binary_image


class Test:

    def __init__(self, path=None, pretrained_model=None):
        self.path = path
        self.input_dir = os.path.join(self.path, "images/")
        self.target_dir = os.path.join(self.path, "annotations/trimaps/")
        self.img_size = (128, 128)
        self.num_classes = 3
        self.batch_size = 128
        self.val_samples = 1000
        self.model = build_compile_model()
        self.model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)

        input_img_paths = sorted([os.path.join(self.input_dir, fname) for fname in os.listdir(self.input_dir)
                                  if fname.endswith(".jpg")])
        target_img_paths = sorted([os.path.join(self.target_dir, fname) for fname in os.listdir(self.target_dir)
                                   if fname.endswith(".png") and not fname.startswith(".")])

        random.Random(1337).shuffle(input_img_paths)
        random.Random(1337).shuffle(target_img_paths)

        self.val_input_img_paths = input_img_paths[-self.val_samples:]
        self.val_target_img_paths = target_img_paths[-self.val_samples:]
        self.valid_dataset = get_dataset_validation(self.batch_size,
                                                    self.val_input_img_paths, self.val_target_img_paths,
                                                    max_dataset_len=None)

    def predict(self):
        # Generate predictions for all images in the validation set
        val_preds = self.model.predict(self.valid_dataset)
        # Display results for first 50 validation images
        for idx in range(1, 50):
            # Display input image
            input_image = np.asarray(cv2.imread(self.val_input_img_paths[idx], cv2.IMREAD_UNCHANGED))
            input_image = resample_image(input_image, shape=self.img_size, interpolation="CUBIC")
            # Display ground-truth target mask
            ground_truth = np.asarray(cv2.imread(self.val_target_img_paths[idx], cv2.IMREAD_UNCHANGED))
            ground_truth = resample_image(ground_truth, shape=self.img_size, interpolation="NEAREST")
            # Display mask predicted by our model
            predicted_mask = np.argmax(val_preds[idx], axis=-1)
            # Post process predicted mask a little
            predicted_mask += 1
            binary_mask = predicted_mask.copy()
            binary_mask[binary_mask == 2] = 0
            binary_mask[binary_mask > 1] = 1
            binary_mask = largest_connected_component_binary_image(binary_image_data=binary_mask)
            predicted_mask[binary_mask == 0] = 2
            output_file = os.path.join(self.path, 'prediction/{}.png'.format(idx))
            save_fig(input_image, ground_truth, predicted_mask, output_file)


if __name__ == '__main__':
    obj = Test(path='/home/saurabh/Desktop/segmentation',
               pretrained_model='/home/saurabh/Desktop/segmentation/models/22-0.286.hdf5')
    obj.predict()
