# Razorback Logo CNN Classifier

This repo contains our Project 3 code for DASC 41103 Machine Learning at the University of Arkansas. The goal is to train a small CNN in PyTorch that can look at an Etsy image and decide if it contains the **official Razorback logo** or not.

## 1. Problem Setup

The university wants to flag Etsy vendors who are using the official Razorback logo without a license. We framed this as a binary image classification task:

- `1` → image contains the official Razorback logo  
- `0` → image does not contain the official Razorback logo

Because this is a class project, we created a small custom dataset instead of using a standard benchmark dataset.

---

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
  - Completely unrelated images (pets, random Etsy items and so on)

**Final counts**

- 30 images with the official logo
- 30 images without the official logo  
  → total of **60 images**

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

---

## 3. Model Architecture

We built a small CNN using `torch.nn.Sequential`. The final architecture is:

1. **Conv block 1**
   - Conv2d: in channels 3, out channels 32, kernel size 3, padding 1  
   - ReLU  
   - MaxPool2d with kernel size 2  
   - Dropout with probability 0.5  

2. **Conv block 2**
   - Conv2d: 32 to 64  
   - ReLU  
   - MaxPool2d with kernel size 2  
   - Dropout with probability 0.5  

3. **Conv block 3**
   - Conv2d: 64 to 128  
   - ReLU  
   - MaxPool2d with kernel size 2  

4. **Conv block 4**
   - Conv2d: 128 to 256  
   - ReLU  

5. **Pooling and classifier**
   - Average pooling over an 8 by 8 window  
   - Flatten  
   - Fully connected layer with 1024 inputs and 1 output  
   - Sigmoid activation to get a probability of “official Razorback logo”

**Loss and optimizer**

- Loss: Binary cross entropy (`nn.BCELoss`)
- Optimizer: Adam with learning rate `0.001`

We chose Adam because it tends to work well out of the box on small datasets and makes training more stable than plain SGD in this kind of setting.

---

## 4. Training, Tuning, and Evaluation

### Hyperparameter choices

We focused on:

- Batch size
- Learning rate
- Number of epochs

We kept the learning rate at `0.001` after trying other values. A larger learning rate around `0.01` was too unstable, while a much smaller one converged very slowly.

We trained for **30 epochs**, watching both training and validation curves. That was about the point where validation accuracy leveled off.

We trained three versions of the model with different batch sizes:

- Batch size 8  
- Batch size 16  
- Batch size 32  

For each run we tracked:

- Training loss and accuracy per epoch  
- Validation loss and accuracy per epoch  

This made it easy to see when the model started to overfit versus when it was learning useful patterns.

### Best model

The **batch size 8** model ended up being our final choice. With such a small dataset, a smaller batch gives more frequent updates and seemed to give better validation performance.

Rough summary of the metrics:

- Validation accuracy peaked around **83 percent**  
- Test accuracy landed around the **mid fifties**, which is not amazing but is reasonable given:
  - Only 9 test images
  - Very small overall dataset

We also generated a classification report for both the validation and test sets to see precision, recall, and F1 for each class, not just overall accuracy.

### Overfitting and regularization

We watched for overfitting by comparing:

- Train loss vs validation loss  
- Train accuracy vs validation accuracy  

Key things we did to keep overfitting under control:

- **Data augmentation**: random horizontal flips and standardization of size  
- **Dropout**: after the first two convolution blocks  
- **Keeping the model fairly small**: not too many layers and a single dense layer at the end

Given how tiny the dataset is, some level of overfitting is almost unavoidable, but these steps helped keep the validation curves from diverging too badly.

---

## 5. Saved Model

The final chosen model is saved as:

```text
models/razorback-cnn.ph
