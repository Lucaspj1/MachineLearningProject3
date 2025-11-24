# Razorback Logo CNN Classifier

This repo contains our Project 3 code for DASC 41103 Machine Learning at the University of Arkansas. The goal is to train a small CNN in PyTorch that can look at an Etsy image and decide if it contains the official Razorback logo or not.

## 1. Problem Setup

The university wants to flag Etsy vendors who are using the official Razorback logo without a license. We framed this as a binary image classification task:

- Class 1: image contains the official Razorback logo  
- Class 0: image does not contain the official Razorback logo  

Because this is a class project, we created a small custom dataset instead of using a standard benchmark dataset.

## 2. Dataset

**Source**

All images were pulled from Etsy using keywords like:

- “Arkansas Razorbacks”
- “Razorbacks”
- plus some generic searches like “dogs”, “cats”, and other random items

**Classes**

- **Positive class (official logo)**  
  Images that contain the official side profile Razorback logo, usually in the correct colors.  
  We allowed a little variation in background and minor color tweaks but stayed close to the official logo shape.

- **Negative class (not official logo)**  
  - Alternate or front facing hog logos  
  - Designs that reference Razorbacks without using the official logo  
  - Completely unrelated images such as pets or other Etsy items

**Final counts**

- 30 images with the official logo
- 30 images without the official logo  
  Total of 60 images in the dataset.

We tried a few different splits at first, including skewed ones with more negatives, but performance was more stable and easier to interpret when we kept the dataset balanced between the two classes.

**Train, validation, test splits**

We used `torch.utils.data.random_split` with a fixed random seed to keep results reproducible:

- Train: 45 images
- Validation: 6 images
- Test: 9 images

Because the dataset is tiny, each individual image matters a lot. We were careful to mix both classes into each split so that the model always saw examples of both “Razorback” and “not Razorback” in training, validation, and testing.

**Transforms**

We used two transform pipelines:

- **Training transforms**
  - Resize shortest side to 128
  - Center crop to 128 by 128
  - Random horizontal flip
  - Convert to tensor

- **Validation and test transforms**
  - Resize to 128
  - Center crop to 128 by 128
  - Convert to tensor

The random horizontal flips act as simple data augmentation and help the model generalize a bit better despite the tiny dataset.

## 3. Model Architecture

We built a small CNN using `torch.nn.Sequential`. The final architecture is:

1. **Conv block 1**  
   Conv2d with 3 input channels and 32 output channels, kernel size 3, padding 1  
   ReLU activation  
   MaxPool2d with kernel size 2  
   Dropout with probability 0.5  

2. **Conv block 2**  
   Conv2d from 32 to 64 channels  
   ReLU activation  
   MaxPool2d with kernel size 2  
   Dropout with probability 0.5  

3. **Conv block 3**  
   Conv2d from 64 to 128 channels  
   ReLU activation  
   MaxPool2d with kernel size 2  

4. **Conv block 4**  
   Conv2d from 128 to 256 channels  
   ReLU activation  

5. **Pooling and classifier**  
   Average pooling over an 8 by 8 window  
   Flatten layer  
   Fully connected layer with 1024 inputs and 1 output  
   Sigmoid activation to get a probability that the image contains the official Razorback logo

**Loss and optimizer**

- Loss: Binary cross entropy (`nn.BCELoss`)  
- Optimizer: Adam with learning rate `0.001`

Adam worked well for this small dataset and gave stable training compared with plain SGD.

## 4. Saved Model

The final chosen model is saved as:

```text
models/razorback-cnn.ph
