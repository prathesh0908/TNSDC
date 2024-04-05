# TNSDC
Description:
This set of code is a Python script for implementing a Convolutional Neural Network (CNN) using the Keras library with TensorFlow backend. The CNN is designed to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The CIFAR-10 dataset is a widely used benchmark for image classification tasks.

Files:
- cifar10_cnn.py: Python script containing the code for building and training the CNN model.
- all_cnn_weights_0.9088_0.4994.hdf5: Pretrained weights file for the CNN model, achieving an accuracy of 90.88% on the CIFAR-10 test set.

Libraries Used:
- Keras: Deep learning library for building neural networks.
- NumPy: Library for numerical computations in Python.
- Matplotlib: Library for creating visualizations in Python.
- PIL (Python Imaging Library): Library for opening, manipulating, and saving many different image file formats.

Instructions:
1. Setup Environment: Ensure that Python and necessary libraries (Keras, NumPy, Matplotlib) are installed in your environment.
2. Dataset: The CIFAR-10 dataset will be automatically downloaded and loaded when running the script.
3. Pretrained Weights (Optional): If you want to use pretrained weights, make sure to have the `all_cnn_weights_0.9088_0.4994.hdf5` file in the same directory as the script.
4. Run Script: Execute the `cifar10_cnn.py` script. This will build the CNN model, load pretrained weights (if provided), compile the model, and evaluate its performance on the test set.
5. Output: The script will display model summary, training progress (if training is enabled), and final accuracy on the test set. Additionally, it will show predictions for a batch of images from the test set along with their corresponding ground truth labels.

Notes:
- The CNN architecture used in this script is based on the All Convolutional Network (All-CNN) model proposed by Springenberg et al. in their paper "Striving for Simplicity: The All Convolutional Net" (2014).
- The provided pretrained weights file (`all_cnn_weights_0.9088_0.4994.hdf5`) is trained on the CIFAR-10 dataset and achieves a classification accuracy of 90.88% on the test set.
- You can modify hyperparameters such as learning rate, weight decay, momentum, and training epochs based on your requirements.
- The script also includes code for visualizing predictions on a batch of images from the test set.

---

This README provides an overview of the contents and usage instructions for the provided Python script and associated files. If you have any further questions or need assistance, feel free to reach out.
