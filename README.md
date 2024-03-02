# UNet
UNet based segmentation using Keras and Python

# Training
1. Download data from : https://www.robots.ox.ac.uk/~vgg/data/pets/. Files are  images.tar.gz (dataset) and annotations.tar.gz (groundtruth data). Unzip both both files in a folder called "segmentation".
2. Open training.py file and provide the "segmentation" folder path.
4. Run python file. Please note, it is assumed you have a proper running enviroment e.g., conda to run the code. Training using GPU is recomemded.

# Testing
1. Open test.py file and provide the data path.
2. Provide path to trained model.
3. In "segmentation" folder, create a folder called "prediction".
4. Run the file.
5. First fifty validation images will be saved in the "prediction" folder.
