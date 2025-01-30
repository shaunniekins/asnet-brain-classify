# v3.1

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

_gen = ImageDataGenerator(
    rescale=1/255,
    brightness_range=(0.8, 1.2),
    rotation_range=10,  
    width_shift_range=0.05,  
    height_shift_range=0.05, 
    shear_range=0.05, 
    zoom_range=0.05,  
    horizontal_flip=True, 
    fill_mode='nearest'
)

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

# Show all images in a dynamic grid (uncomment to use)
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
    
    # Create the learning rate scheduler
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,  # Initial learning rate for the scheduler
        decay_steps=36,
        decay_rate=0.98, 
        staircase=True
    )

    # Create Adam optimizer with the learning rate scheduler
    optimizer = Adam(learning_rate=lr_scheduler)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall'],  # Use string identifiers instead of metric instances
        steps_per_execution=32  # Added steps_per_execution for TPU optimization
    )

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)


# 4. Training
num_epochs = 20

# Add TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=1, write_graph=True, profile_batch=0)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

hist = model.fit(
    tr_gen,
    epochs=num_epochs,
    validation_data=valid_gen,
    shuffle=False,
    callbacks=[early_stopping, tensorboard_callback]  # Add the early stopping callback
)

"""
Epoch 1/20
179/179 [==============================] - 179s 997ms/step - loss: 0.9196 - accuracy: 0.6331 - precision: 0.8198 - recall: 0.4007 - val_loss: 0.8384 - val_accuracy: 0.7053 - val_precision: 0.7478 - val_recall: 0.6519
Epoch 2/20
179/179 [==============================] - 168s 936ms/step - loss: 0.4631 - accuracy: 0.8405 - precision: 0.8879 - recall: 0.7778 - val_loss: 0.5082 - val_accuracy: 0.8290 - val_precision: 0.8368 - val_recall: 0.8061
Epoch 3/20
179/179 [==============================] - 168s 938ms/step - loss: 0.3673 - accuracy: 0.8797 - precision: 0.9065 - recall: 0.8440 - val_loss: 0.4638 - val_accuracy: 0.8580 - val_precision: 0.8636 - val_recall: 0.8412
Epoch 4/20
179/179 [==============================] - 168s 941ms/step - loss: 0.3152 - accuracy: 0.8985 - precision: 0.9179 - recall: 0.8694 - val_loss: 0.4090 - val_accuracy: 0.8412 - val_precision: 0.8465 - val_recall: 0.8336
Epoch 5/20
179/179 [==============================] - 166s 929ms/step - loss: 0.2819 - accuracy: 0.9107 - precision: 0.9256 - recall: 0.8885 - val_loss: 0.4373 - val_accuracy: 0.8473 - val_precision: 0.8777 - val_recall: 0.8214
Epoch 6/20
179/179 [==============================] - 167s 935ms/step - loss: 0.2467 - accuracy: 0.9181 - precision: 0.9351 - recall: 0.9004 - val_loss: 0.3372 - val_accuracy: 0.8809 - val_precision: 0.8872 - val_recall: 0.8763
Epoch 7/20
179/179 [==============================] - 165s 924ms/step - loss: 0.2244 - accuracy: 0.9272 - precision: 0.9431 - recall: 0.9161 - val_loss: 0.3648 - val_accuracy: 0.8733 - val_precision: 0.9010 - val_recall: 0.8473
Epoch 8/20
179/179 [==============================] - 163s 910ms/step - loss: 0.2047 - accuracy: 0.9326 - precision: 0.9438 - recall: 0.9202 - val_loss: 0.2998 - val_accuracy: 0.8992 - val_precision: 0.9062 - val_recall: 0.8855
Epoch 9/20
179/179 [==============================] - 164s 915ms/step - loss: 0.1930 - accuracy: 0.9377 - precision: 0.9489 - recall: 0.9270 - val_loss: 0.3560 - val_accuracy: 0.8748 - val_precision: 0.8824 - val_recall: 0.8702
Epoch 10/20
179/179 [==============================] - 166s 930ms/step - loss: 0.1839 - accuracy: 0.9431 - precision: 0.9517 - recall: 0.9279 - val_loss: 0.2332 - val_accuracy: 0.9191 - val_precision: 0.9253 - val_recall: 0.9084
Epoch 11/20
179/179 [==============================] - 165s 921ms/step - loss: 0.1679 - accuracy: 0.9477 - precision: 0.9571 - recall: 0.9377 - val_loss: 0.2171 - val_accuracy: 0.9252 - val_precision: 0.9297 - val_recall: 0.9084
Epoch 12/20
179/179 [==============================] - 165s 924ms/step - loss: 0.1634 - accuracy: 0.9505 - precision: 0.9593 - recall: 0.9414 - val_loss: 0.2254 - val_accuracy: 0.9206 - val_precision: 0.9275 - val_recall: 0.9176
Epoch 13/20
179/179 [==============================] - 165s 919ms/step - loss: 0.1485 - accuracy: 0.9562 - precision: 0.9638 - recall: 0.9452 - val_loss: 0.2594 - val_accuracy: 0.9099 - val_precision: 0.9112 - val_recall: 0.9084
Epoch 14/20
179/179 [==============================] - 164s 919ms/step - loss: 0.1424 - accuracy: 0.9564 - precision: 0.9636 - recall: 0.9510 - val_loss: 0.2354 - val_accuracy: 0.9176 - val_precision: 0.9199 - val_recall: 0.9115
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
Train Loss: 0.1131
Train Accuracy: 96.29%
--------------------
Validation Loss: 0.2507
Validation Accuracy: 90.99%
--------------------
Test Loss: 0.1937
Test Accuracy: 93.60%
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

           0       0.97      0.93      0.95       150
           1       0.95      0.82      0.88       153
           2       0.94      0.99      0.96       203
           3       0.89      1.00      0.94       150

    accuracy                           0.94       656
   macro avg       0.94      0.93      0.93       656
weighted avg       0.94      0.94      0.93       656
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
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/meningioma/Te-me_0010.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/meningioma/Te-me_0030.jpg')
# it predicted "pituitary" with 1.00 probability


predict('/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0007.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-gl_0017.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/glioma/Te-gl_0037.jpg')
# it predicted "pituitary" with 1.00 probability


predict('/kaggle/input/brain-tumor-mri-dataset/Testing/notumor/Te-noTr_0001.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/notumor/Te-no_0011.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/notumor/Te-no_0031.jpg')
# it predicted "pituitary" with 1.00 probability


predict('/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0001.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0011.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0031.jpg')
# it predicted "pituitary" with 1.00 probability