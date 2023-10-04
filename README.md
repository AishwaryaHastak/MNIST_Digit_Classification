# MNIST_Handwritten_Digit_Image_Classification

Image Classification of handwritten digits in the MNIST dataset using CNN

Modified the architecture of the existing Lenet-5 model.

![image](https://github.com/AishwaryaHastak/MNIST_Digit_Classification/assets/31357026/8a42ecdd-b34d-4b5c-a14b-09cc6996d786)


The best achieved accuracy of the modified model is 98.49%

# UNDERSTANDING OF THE PROBLEM
The problem is a multi-class image classification problem, where we have to identify the handwritten digit in the image.
To do this job, we will build a deep neural net model with multiple convolution networks, pooling layers, and fully connected layers. We are making use of the famous LENET architecture for MNIST dataset image classification.

---

# THOUGHT BEHIND THE CODE

There are four main parts to the code:

**Loading the Dataset**: The images are of size 28*28 pixels. We transform all images to a common size of 32*32 to be able to implement the LENET architecture.  We load the dataset available in the pytorch datasets class and define the training and testing dataloader.

**Defining the model**: We define a class for a custom model that inherits nn.Module, which allows for automatic parameter initialization and tracking and gradient computation. We define the layers in the initialization method and define the forward pass.

**Training**: We set the model in training mode using model.train(). We iterate over each batch in the training dataloader for n number of epochs, setting the gradients to zero at the beginning of each batch processing, getting the model outputs, calculating loss, and performing a backward pass to calculate the gradients, and then finally updating the model parameters. We calculate and record the loss and accuracy after each epoch.

**Testing**: We set the model in evaluation mode using model.eval() [which is also a functionality provided by the nn.Module class]. We iterate over the testing dataloader and get the testing loss and accuracy for each epoch.

---
# OUTPUTS

### INITIAL MODEL

The initial model with the standard LENET architecture and a batch size of 600 and 6 epochs produced an accuracy of 97%.

![image](https://github.com/AishwaryaHastak/MNIST_Digit_Classification/assets/31357026/66cea2a6-3390-444b-8e62-0872293c917c)


### USING DIFFERENT NUMBER OF CONVOLUTION LAYERS

![image](https://github.com/AishwaryaHastak/MNIST_Digit_Classification/assets/31357026/62157475-7bb1-43b9-abe9-1b7a71a6755e)


Added a fourth Conv2d layer with 50 output features, and used tanh activation function and maxpool function for the first layer. This model performed much better and produced an accuracy of 98.26%.

### USING DIFFERENT NUMBER OF FULLY CONNECTED LAYERS

Using 2 FC Layers, This is performed the best out of all model changes with an accuracy of 98.49%.

![image](https://github.com/AishwaryaHastak/MNIST_Digit_Classification/assets/31357026/7dd9676e-0dd5-4488-adea-aa2ef3cc8e4f)

---

# FINDINGS

The best performing model achieved an accuracy of 98.49%, with an additional fourth Conv2d layer with 50 output features, and used tanh activation function and maxpool function for the first layer, and two fully connected layers instead of three (as in the original LENET model architecture).

The better performance by increasing the number of convolution layers could be attributed to increasing the input channels and the number of convolution layers that seemed to have helped the model make more accurate prediction.

Increasing the number of convolution layers (depth of the model) and the number of channels (width of the model) leads to capturing more complex patterns in the data; this could be because of the model having more data points to work with.

This could also lead to overfitting. Hence, the balance has to be carefully maintained. Decreasing the number of fully connected layers can also help in increasing the generalizability of the model by making it focus more on the essential parts of the data.


