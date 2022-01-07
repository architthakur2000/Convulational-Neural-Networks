# Convulational-Neural-Networks
In this project, I designed a simple CNN model along with my own convolutional network for a more realistic dataset -- MiniPlaces, again using PyTorch. Dataset  MiniPlaces is a scene recognition dataset developed by MIT. This dataset has 120K images from 100 scene categories. The categories are mutually exclusive. The dataset is split into 100K images for training, 10K images for validation, and 10K for testing.  

Helper Codes (train_miniplaces.py and dataloader.py) exist (See the comments in these files for more implementation details). The original image resolution for images in MiniPlaces is 128x128. To make the training feasible, the data loader reduces the image resolution to 32x32. One can always assume this input resolution. Our data loader will also download the full dataset the first time you run train_miniplaces.py  Before the training procedure, we define the dataloader, model, optimizer, image transform and criterion. Execution of the training and testing in function train_model and test_model. 

This implementation takes one or two hours to run. In this project I explored building deep neural networks, including Convolutional Neural Networks (CNNs), using PyTorch. 

The data set is not included since it uses web-scraping. 
