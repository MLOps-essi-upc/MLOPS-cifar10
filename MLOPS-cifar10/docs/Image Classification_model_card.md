---

---






# Model Card for Image Classification

<!-- Provide a quick summary of what the model is/does. [Optional] -->
The goal of this model is to classify a given image into one of 10 classes from cifar10 dataset (airplane, automobile, bird, cat, frog, deer, dog, horse, ship or truck) with an accuracy of at least 90%.




#  Table of Contents

- [Model Card for Image Classification](#model-card-for--model_id-)
- [Table of Contents](#table-of-contents)
- [Table of Contents](#table-of-contents-1)
- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
  - [Direct Use](#direct-use)
  - [Downstream Use [Optional]](#downstream-use-optional)
  - [Out-of-Scope Use](#out-of-scope-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  - [Recommendations](#recommendations)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
    - [Preprocessing](#preprocessing)
    - [Speeds, Sizes, Times](#speeds-sizes-times)
- [Evaluation](#evaluation)
  - [Testing Data, Factors & Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Factors](#factors)
    - [Metrics](#metrics)
  - [Results](#results)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications [optional]](#technical-specifications-optional)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Hardware](#hardware)
    - [Software](#software)
- [Citation](#citation)
- [Glossary [optional]](#glossary-optional)
- [More Information [optional]](#more-information-optional)
- [Model Card Authors [optional]](#model-card-authors-optional)
- [Model Card Contact](#model-card-contact)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)


# Model Details

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
The goal of this model is to classify a given image into one of 10 different classes with an accuracy of at least 90%.

- **Developed by:** Leandra Moonsammy, Dorota Solarska, David Carela, Edu Masip
- **Shared by [Optional]:** Leandra Moonsammy, Dorota Solarska, David Carela, Edu Masip
- **Model type:** Deep Learning
- **Language(s) (NLP):** en
- **License:** mit
- **Parent Model:** [Google's ViT model pre-trained on ImageNet-21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- **Resources for more information:** More information needed
    - [GitHub Repo](https://github.com/MLOps-essi-upc/cifar10.git)


# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

## Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->




## Downstream Use [Optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->
 



## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." --> 

This model is only intended to be used for classifying images with the previous 10 classes mentioned: airplane, automobile, bird, cat, frog, deer, dog, horse, ship or truck. 
Users should not use the model to classify or generate inappropriate or explicit content. 
Users should not deploy the model for surveillance or privacy-invasive purposes without consent.
Users should not use the model to generate copyrighted or trademarked material without proper authorization.



# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model may exhibit bias towards certain classes or underrepresented groups in the CIFAR-10 dataset, leading to unfair predictions.


## Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Perform a comprehensive fairness assessment, monitor model performance across different subgroups, and address bias in data collection and pre-processing.



# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.


## Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

### Preprocessing

More information needed

### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

The dataset is divided into five training batches, each with 10000 images. 
 
# Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

## Testing Data, Factors & Metrics

### Testing Data

<!-- This should link to a Data Card if possible. -->

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 10000 test images. The dataset is divided into one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class.


### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

The dataset contains an equal number of images per class, but training batches may have variations in class distribution. 
CIFAR-10 images are small (32x32 pixels), which may affect the model's ability to distinguish fine details.
The dataset may contain noisy or mislabeled images.


### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Accuracy is a fundamental metric that measures the ratio of correctly classified samples to the total number of samples. It provides an overall view of the model's performance but doesn't consider class imbalances.
The F1-score is the harmonic mean of precision and recall, providing a balance between these two metrics. It is especially useful when there are trade-offs between false positives and false negatives.

## Results 

More information needed

# Model Examination

More information needed

# Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** More information needed
- **Hours used:** More information needed
- **Cloud Provider:** AWS
- **Compute Region:** EMEA
- **Carbon Emitted:** More information needed

# Technical Specifications [optional]

## Model Architecture and Objective

More information needed

## Compute Infrastructure

More information needed

### Hardware

More information needed

### Software

More information needed

# Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

More information needed

**APA:**

More information needed

# Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

More information needed

# More Information [optional]

More information needed

# Model Card Authors [optional]

<!-- This section provides another layer of transparency and accountability. Whose views is this model card representing? How many voices were included in its construction? Etc. -->

Leandra Moonsammy, Dorota Solarska, David Carela, Edu Masip

# Model Card Contact

More information needed

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

More information needed

</details>
