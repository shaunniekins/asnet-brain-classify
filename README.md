# AS-Net for Brain Tumor MRI Classification: Comparing Standard and Lightweight Encoders

This implementation compares three encoder backbones—VGG16, MobileNetV3Large, and EfficientNetV2B0—to assess their impact on classification performance and efficiency.

The original AS-Net paper focuses on skin lesion segmentation and can be found [here](https://doi.org/10.1016/j.eswa.2022.117112).

## Project Structure

The project includes three AS-Net implementations, each using a different encoder: VGG16, MobileNetV3Large, and EfficientNetV2B0.
Each implementation is provided as both a Python script (.py) and a Jupyter notebook (.ipynb) for flexible development and experimentation.

## Datasets

The project uses the Brain Tumor MRI Dataset available on Kaggle, uploaded by Masoud Nickparvar:
<https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

The dataset contains MRI images categorized into four classes:

- Meningioma
- Glioma
- Pituitary tumor
- No tumor (normal)

## Setup and Process

### Requirements

- Python 3.8+
- TensorFlow 2.5+
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Pillow)

### Workflow

1. **Data Preprocessing**:
   - Loading and organizing data into DataFrames
   - Train/validation/test splitting
   - Data augmentation (rotation, zoom, flip, etc.)

2. **Model Architecture**:
   - Encoder backbone (VGG16, MobileNetV3Large, or EfficientNetV2B0)
   - Spatial Attention Module (SAM)
   - Channel Attention Module (CAM)
   - Synergy Module to combine attention features
   - Classification head

3. **Training**:
   - Transfer learning with pretrained encoders
   - Learning rate scheduling
   - Class weight balancing
   - Early stopping

4. **Evaluation**:
   - Accuracy, precision, recall metrics
   - Confusion matrices
   - Classification reports
   - Test-time augmentation

## Model Performance

After evaluation, the following metrics were computed across the different encoder models:

### VGG16 Encoder

- Test Accuracy: 99.24%
- Macro F1-Score: 0.99
- Training Time: ~35 min

### MobileNetV3Large Encoder

- Test Accuracy: 97.26%
- Macro F1-Score: 0.97
- Training Time: ~27 min

### EfficientNetV2B0 Encoder

- Test Accuracy: 78.66%
- Macro F1-Score: 0.77
- Training Time: ~32 min

### Comparison Observations

- **MobileNetV3Large** achieved the highest accuracy despite being a lightweight model, demonstrating the effectiveness of the AS-Net architecture when paired with modern efficient encoders.
- **VGG16** performed well but required longer training time due to its larger parameter count.
- **EfficientNetV2B0** had lower performance, possibly requiring more tuning specifically for medical imaging applications.

The integration of spatial and channel attention mechanisms helped all models to focus on relevant tumor regions in MRI scans, with the lightweight MobileNetV3Large proving to be the most effective combination of accuracy and efficiency.
