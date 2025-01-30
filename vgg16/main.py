# v1

"""
!apt-get update
!apt-get install graphviz -y

!pip install --upgrade pip
!pip install graphviz
!pip install seaborn
!pip install pydot
"""


# 1. Import needed libraries
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
# ---------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# ---------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Conv2D, concatenate, Multiply, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
# ---------------------------------------
import warnings
warnings.filterwarnings("ignore")

# Detect and initialize the TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None
    print("No TPU detected. Running on CPU/GPU")
    
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    tpu_strategy = tf.distribute.get_strategy()
    
print("REPLICAS: ", tpu_strategy.num_replicas_in_sync)


# Preprocessing

## 2.1 Load Data
def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])

    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])

    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df

tr_df = train_df('/kaggle/input/brain-tumor-mri-dataset/Training')
ts_df = test_df('/kaggle/input/brain-tumor-mri-dataset/Testing')

# Count of images in each class in train data
plt.figure(figsize=(15,7))
ax = sns.countplot(data=tr_df , y=tr_df['Class'])

plt.xlabel('')
plt.ylabel('')
plt.title('Count of images in each class', fontsize=20)
ax.bar_label(ax.containers[0])
plt.show()

#Count each class in test data
plt.figure(figsize=(15, 7))
ax = sns.countplot(y=ts_df['Class'], palette='viridis')

ax.set(xlabel='', ylabel='', title='Count of images in each class')
ax.bar_label(ax.containers[0])

plt.show()


## 2.2 Split data into train, test, valid
valid_df, ts_df = train_test_split(ts_df, train_size=0.5, random_state=20, stratify=ts_df['Class'])

## 2.3 Data preprocessing
BATCH_SIZE = 32 * tpu_strategy.num_replicas_in_sync  # Scales with TPU cores
IMAGE_SIZE = (299, 299)

_gen = ImageDataGenerator(rescale=1/255,
                          brightness_range=(0.8, 1.2))

ts_gen = ImageDataGenerator(rescale=1/255)


tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path',
                                 y_col='Class', batch_size=BATCH_SIZE,
                                 target_size=IMAGE_SIZE)

valid_gen = _gen.flow_from_dataframe(valid_df, x_col='Class Path',
                                    y_col='Class', batch_size=BATCH_SIZE,
                                    target_size=IMAGE_SIZE)

ts_gen = ts_gen.flow_from_dataframe(ts_df, x_col='Class Path',
                                   y_col='Class', batch_size=BATCH_SIZE,
                                   target_size=IMAGE_SIZE, shuffle=False)

## 2.4 Getting samples from data
# Get the class dictionary and classes list
class_dict = tr_gen.class_indices
classes = list(class_dict.keys())

# Get a batch of images
images, labels = next(ts_gen)

# Calculate grid dimensions based on number of images
n_images = len(images)
grid_size = int(np.ceil(np.sqrt(n_images)))  # Make a square grid

# Create the plot
plt.figure(figsize=(20, 20))

# Option 1: Show only the first 16 images
# for i, (image, label) in enumerate(zip(images[:16], labels[:16])):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(image)
#     class_name = classes[np.argmax(label)]
#     plt.title(class_name, color='k', fontsize=15)
#     plt.axis('off')

# Option 2: Show all images in a dynamic grid (uncomment to use)
for i, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(image)
    class_name = classes[np.argmax(label)]
    plt.title(class_name, color='k', fontsize=15)
    plt.axis('off')

plt.tight_layout()
plt.show()


# 3. Building Deep Learning Model
class SAM(Model):
    def __init__(self, filters):
        super(SAM, self).__init__()
        self.filters = filters
        self.conv1 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(self.filters // 4, 1,
                            activation='relu', kernel_initializer='he_normal')
        self.W1 = Conv2D(self.filters // 4, 1,
                         activation='sigmoid', kernel_initializer='he_normal')
        self.W2 = Conv2D(self.filters // 4, 1,
                         activation='sigmoid', kernel_initializer='he_normal')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)

        pool1 = GlobalAveragePooling2D()(out2)
        pool1 = Reshape((1, 1, self.filters // 4))(pool1)
        merge1 = self.W1(pool1)

        pool2 = GlobalMaxPooling2D()(out2)
        pool2 = Reshape((1, 1, self.filters // 4))(pool2)
        merge2 = self.W2(pool2)

        out3 = merge1 + merge2
        y = Multiply()([out1, out3]) + out2
        return y


class CAM(Model):
    def __init__(self, filters, reduction_ratio=16):
        super(CAM, self).__init__()
        self.filters = filters
        self.conv1 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(self.filters // 4, 1,
                            activation='relu', kernel_initializer='he_normal')
        self.gpool = GlobalAveragePooling2D()
        self.fc1 = Dense(self.filters // (4 * reduction_ratio),
                         activation='relu', use_bias=False)
        self.fc2 = Dense(self.filters // 4,
                         activation='sigmoid', use_bias=False)

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)
        out3 = self.fc2(self.fc1(self.gpool(out2)))
        out3 = Reshape((1, 1, self.filters // 4))(out3)
        y = Multiply()([out1, out3]) + out2
        return y


class ResizeLayer(Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, (self.target_height, self.target_width))


def adjust_feature_map(x, target_shape):
    _, h, w, _ = target_shape
    current_h, current_w = x.shape[1:3]
    if current_h != h or current_w != w:
        resize_layer = ResizeLayer(h, w)
        return resize_layer(x)
    return x


# Modified AS_Net with VGG16 encoder
def AS_Net(encoder='vgg16', input_size=(299, 299, 3), fine_tune_at=None):
    inputs = Input(input_size)
    print(f'CURRENT ENCODER: {encoder}')

    if encoder == 'vgg16':
        # Load VGG16 with ImageNet weights
        ENCODER = VGG16(weights='imagenet', include_top=False, input_shape=input_size)

        # Freeze all layers initially
        ENCODER.trainable = False

        # Optionally, unfreeze layers for fine-tuning from a certain layer
        if fine_tune_at is not None:
            for layer in ENCODER.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in ENCODER.layers[fine_tune_at:]:
                layer.trainable = True

        # Selected output layers (you can experiment with different indices)
        layer_indices = [2, 5, 9, 13, 17]
    else:
        raise ValueError("Unsupported encoder type. Only 'vgg16' is supported in this case.")

    # Get the output layers dynamically
    output_layers = [ENCODER.get_layer(index=i).output for i in layer_indices]
    outputs = [Model(inputs=ENCODER.inputs, outputs=layer)(inputs)
               for layer in output_layers]

    # Adjust and merge feature maps
    merged = outputs[-1]
    for i in range(len(outputs) - 2, -1, -1):
        adjusted = adjust_feature_map(outputs[i], merged.shape)
        merged = concatenate([merged, adjusted], axis=-1)

    # Apply SAM and CAM, scale filters dynamically based on merged feature size
    filters = merged.shape[-1]
    SAM1 = SAM(filters=filters)(merged)
    CAM1 = CAM(filters=filters)(merged)

    # Combine SAM and CAM outputs
    combined = concatenate([SAM1, CAM1], axis=-1)

    # Final classification layers (added more layers and dropout)
    final_layers = Sequential([
        Conv2D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Flatten(),
        Dropout(0.5),  # Increased dropout to prevent overfitting
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(4, activation='softmax', kernel_initializer='he_normal')  # 4 classes, adjust if needed
    ])

    output = final_layers(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

# Create and compile the model
with tpu_strategy.scope():
    # Create and compile the model
    model = AS_Net(encoder='vgg16', fine_tune_at=15)
    
    # Compile the model (only once)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall'],  # Use string identifiers instead of metric instances
        steps_per_execution=32  # Added steps_per_execution for TPU optimization
    )

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)


# 4. Training
num_epochs = 25
hist = model.fit(tr_gen,
                 epochs=num_epochs,
                 validation_data=valid_gen,
                 shuffle=False)

"""
Epoch 1/25
357/357 [==============================] - 105s 295ms/step - loss: 0.4890 - accuracy: 0.8227 - precision: 0.8722 - recall: 0.7600 - val_loss: 0.5377 - val_accuracy: 0.8427 - val_precision: 0.8610 - val_recall: 0.8321
Epoch 2/25
357/357 [==============================] - 95s 265ms/step - loss: 0.2532 - accuracy: 0.9116 - precision: 0.9257 - recall: 0.9002 - val_loss: 1.6921 - val_accuracy: 0.6260 - val_precision: 0.6263 - val_recall: 0.6244
Epoch 3/25
357/357 [==============================] - 95s 265ms/step - loss: 0.1808 - accuracy: 0.9396 - precision: 0.9454 - recall: 0.9310 - val_loss: 0.3123 - val_accuracy: 0.8901 - val_precision: 0.8995 - val_recall: 0.8885
Epoch 4/25
357/357 [==============================] - 95s 266ms/step - loss: 0.1211 - accuracy: 0.9611 - precision: 0.9649 - recall: 0.9573 - val_loss: 0.2849 - val_accuracy: 0.9160 - val_precision: 0.9240 - val_recall: 0.9099
Epoch 5/25
357/357 [==============================] - 95s 266ms/step - loss: 0.1083 - accuracy: 0.9645 - precision: 0.9674 - recall: 0.9601 - val_loss: 0.3223 - val_accuracy: 0.9206 - val_precision: 0.9377 - val_recall: 0.9191
Epoch 6/25
357/357 [==============================] - 95s 265ms/step - loss: 0.0772 - accuracy: 0.9750 - precision: 0.9775 - recall: 0.9720 - val_loss: 0.5369 - val_accuracy: 0.8046 - val_precision: 0.8111 - val_recall: 0.8000
Epoch 7/25
357/357 [==============================] - 95s 265ms/step - loss: 0.0695 - accuracy: 0.9793 - precision: 0.9816 - recall: 0.9785 - val_loss: 0.7658 - val_accuracy: 0.7802 - val_precision: 0.7819 - val_recall: 0.7771
Epoch 8/25
357/357 [==============================] - 95s 265ms/step - loss: 0.0652 - accuracy: 0.9795 - precision: 0.9812 - recall: 0.9778 - val_loss: 0.5317 - val_accuracy: 0.8733 - val_precision: 0.8731 - val_recall: 0.8718
Epoch 9/25
357/357 [==============================] - 95s 265ms/step - loss: 0.0456 - accuracy: 0.9853 - precision: 0.9861 - recall: 0.9841 - val_loss: 0.0777 - val_accuracy: 0.9756 - val_precision: 0.9756 - val_recall: 0.9756
Epoch 10/25
357/357 [==============================] - 95s 266ms/step - loss: 0.0470 - accuracy: 0.9853 - precision: 0.9858 - recall: 0.9839 - val_loss: 0.3544 - val_accuracy: 0.9267 - val_precision: 0.9267 - val_recall: 0.9267
...
Epoch 24/25
357/357 [==============================] - 95s 266ms/step - loss: 0.0316 - accuracy: 0.9881 - precision: 0.9886 - recall: 0.9879 - val_loss: 0.3443 - val_accuracy: 0.9115 - val_precision: 0.9177 - val_recall: 0.9023
Epoch 25/25
357/357 [==============================] - 95s 265ms/step - loss: 0.0227 - accuracy: 0.9926 - precision: 0.9930 - recall: 0.9925 - val_loss: 0.1348 - val_accuracy: 0.9725 - val_precision: 0.9740 - val_recall: 0.9710
"""

hist.history.keys()

## 4.1 Visualize model performance
tr_acc = hist.history['accuracy']
tr_loss = hist.history['loss']
tr_per = hist.history['precision']
tr_recall = hist.history['recall']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']
val_per = hist.history['val_precision']
val_recall = hist.history['val_recall']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
index_precision = np.argmax(val_per)
per_highest = val_per[index_precision]
index_recall = np.argmax(val_recall)
recall_highest = val_recall[index_recall]

Epochs = [i + 1 for i in range(len(tr_acc))]
loss_label = f'Best epoch = {str(index_loss + 1)}'
acc_label = f'Best epoch = {str(index_acc + 1)}'
per_label = f'Best epoch = {str(index_precision + 1)}'
recall_label = f'Best epoch = {str(index_recall + 1)}'


plt.figure(figsize=(20, 12))
plt.style.use('fivethirtyeight')


plt.subplot(2, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(Epochs, tr_per, 'r', label='Precision')
plt.plot(Epochs, val_per, 'g', label='Validation Precision')
plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
plt.title('Precision and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(Epochs, tr_recall, 'r', label='Recall')
plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
plt.title('Recall and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.show()

# 5. Testing and Evaluation
## 5.1 Evaluate

train_score = model.evaluate(tr_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(ts_gen, verbose=1)

print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")
print('-' * 20)
print(f"Validation Loss: {valid_score[0]:.4f}")
print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
print('-' * 20)
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")


"""
Train Loss: 0.0027
Train Accuracy: 99.91%
--------------------
Validation Loss: 0.1006
Validation Accuracy: 97.71%
--------------------
Test Loss: 0.1041
Test Accuracy: 97.71%
"""

preds = model.predict(ts_gen)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(ts_gen.classes, y_pred)
labels = list(class_dict.keys())
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('Truth Label')
plt.show()

clr = classification_report(ts_gen.classes, y_pred)
print(clr)

"""
              precision    recall  f1-score   support

           0       0.99      0.95      0.97       150
           1       0.95      0.95      0.95       153
           2       1.00      1.00      1.00       203
           3       0.96      1.00      0.98       150

    accuracy                           0.98       656
   macro avg       0.98      0.98      0.98       656
weighted avg       0.98      0.98      0.98       656
"""

## 5.2 Testing
def predict(img_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    label = list(class_dict.keys())
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    predictions = model.predict(img)
    probs = list(predictions[0])
    labels = label
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt = '%.2f')
    plt.show()

predict('/kaggle/input/brain-tumor-mri-dataset/Testing/meningioma/Te-meTr_0000.jpg')
# it predicted "glioma" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0007.jpg')
# it predicted "glioma" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/notumor/Te-noTr_0001.jpg')
# it predicted "glioma" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0001.jpg')
# it predicted "glioma" with 1.00 probability