# SWE3052-Image-Classification-Kaggle
This repository is for final project done in SWE3052 Introduction to Deep Neural Network by Prof. Jongwook Lee in Spring 2021. The goal of the project is to build a semi-supervised machine learning model for image classification using Pytorch, where a few data are only labeled and most of the data are unlabeled. The competition was held in [Kaggle](https://www.kaggle.com/competitions/dnn2021ssl/overview)

## Implementation 
### Pseudo Labeling
- The steps of my pseudo labeled training is as follows:
  1. Train the model with labeled data until convergence or early stopping triggered
  2. Infer on unlabeled data using the trained model
  3. Use the predictions gotten from ii) as the pseudo labels of the data
  4. Calculate the loss and adjust the loss by the parameter alpha ( loss = loss * a)
  5. Backpropagation
  6. [every 50 iteration] Train one epoch on labeled train dataset
  7. Repeat ii) to vi) until convergence (or early stopping triggered)
  
### Data Augmentation
- Used data augmentation for the labeled training dataset
- Used two data augmentation methods; random horizontal flip and random crop.

### Data Preprocessing
- Utilized 4500 images for traing, 500 images for validation
- Resized, tensorized and normalized the images

### Model 
- Used pretrained ResNet50 model 
- Since there is not much data that for training, it is better to finetune the pretrained model rather than training the whole model from scratch
 
 ### Hyperparameters
|    Parmeter   |      Value    |
| ------------- | ------------- |
|    Momentum   |       0.9     |
|  Weight Decay |      0.005    |
| Learning Rate |      0.001    |
|  Train Batch  |      256      |
|  Test Batch   |      256      |

## Result
Scored **92.8%** accuracy on 10,000 test data and ranked **7th** place out of 42 participants.
![image](https://user-images.githubusercontent.com/28348839/198411976-0aff97d3-7028-401c-9df9-26947fa692db.png)

