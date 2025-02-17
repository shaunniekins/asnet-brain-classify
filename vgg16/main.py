# vgg16/main.py
# v5


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
# ---------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
# ---------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv2D, concatenate, Multiply, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
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

tr_df = train_df('/kaggle/input/Training')
ts_df = test_df('/kaggle/input/Testing')

# Count of images in each class in train data
plt.figure(figsize=(15,7))
ax = sns.countplot(data=tr_df , y=tr_df['Class'])

plt.xlabel('')
plt.ylabel('')
plt.title('Count of images in each class', fontsize=20)
ax.bar_label(ax.containers[0])

plt.show()

# Count each class in test data
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

def prepare_data(tr_df, ts_df):
    def process_image(file_path):
        img = Image.open(file_path)
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMAGE_SIZE)
        return np.array(img)

    # Convert DataFrame to numpy arrays
    X = np.array([process_image(f) for f in tr_df['Class Path']], dtype=np.float32) / 255.0
    y = pd.get_dummies(tr_df['Class']).values
    
    X_test = np.array([process_image(f) for f in ts_df['Class Path']], dtype=np.float32) / 255.0
    y_test = pd.get_dummies(ts_df['Class']).values
    
    return X, y, X_test, y_test


def get_augmentation():
    return ImageDataGenerator(
        rescale=1/255,
        brightness_range=(0.9, 1.1),
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect'
    )

_gen = get_augmentation()

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


# AS_Net with VGG16 encoder
def AS_Net(encoder='vgg16', input_size=(299, 299, 3), fine_tune_at=None, reg_factor=0.0005):  # Reduced reg_factor
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

    # Simplify the final layers
    final_layers = Sequential([
        Conv2D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    output = final_layers(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

# Create and compile the model
with tpu_strategy.scope():
    model = AS_Net(encoder='vgg16', fine_tune_at=12)
    
    # Simplified learning rate setup
    initial_learning_rate = 1e-4
    
    optimizer = Adam(learning_rate=initial_learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)


# 4. Training
num_epochs = 35

# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', 
    histogram_freq=1, 
    write_graph=True, 
    profile_batch=0
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Compute class weights
def get_class_weights(y):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(class_weights))

# Calculate balanced class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(tr_gen.classes),
    y=tr_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))

# Use in training
hist = model.fit(
    tr_gen,
    epochs=num_epochs,
    validation_data=valid_gen,
    shuffle=True,
    class_weight=class_weight_dict,  # Use computed class weights
    callbacks=[
        early_stopping,
        tensorboard_callback,
        reduce_lr,
        checkpoint
    ]
)


"""
Epoch 1/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 255s 1s/step - accuracy: 0.7615 - auc: 0.9279 - loss: 0.6486 - precision: 0.8527 - recall: 0.6257 - val_accuracy: 0.6504 - val_auc: 0.8446 - val_loss: 1.5510 - val_precision: 0.6558 - val_recall: 0.6427 - learning_rate: 1.0000e-04
Epoch 2/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 161s 869ms/step - accuracy: 0.9049 - auc: 0.9859 - loss: 0.2814 - precision: 0.9184 - recall: 0.8950 - val_accuracy: 0.8519 - val_auc: 0.9678 - val_loss: 0.4729 - val_precision: 0.8571 - val_recall: 0.8427 - learning_rate: 1.0000e-04
Epoch 3/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 160s 863ms/step - accuracy: 0.9244 - auc: 0.9909 - loss: 0.2202 - precision: 0.9345 - recall: 0.9147 - val_accuracy: 0.8885 - val_auc: 0.9881 - val_loss: 0.2606 - val_precision: 0.8937 - val_recall: 0.8855 - learning_rate: 1.0000e-04
Epoch 4/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 158s 852ms/step - accuracy: 0.9521 - auc: 0.9947 - loss: 0.1525 - precision: 0.9572 - recall: 0.9475 - val_accuracy: 0.7359 - val_auc: 0.9079 - val_loss: 0.9863 - val_precision: 0.7358 - val_recall: 0.7313 - learning_rate: 1.0000e-04
Epoch 5/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 163s 879ms/step - accuracy: 0.9560 - auc: 0.9957 - loss: 0.1355 - precision: 0.9616 - recall: 0.9506 - val_accuracy: 0.9267 - val_auc: 0.9927 - val_loss: 0.2075 - val_precision: 0.9361 - val_recall: 0.9176 - learning_rate: 1.0000e-04
Epoch 6/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 161s 869ms/step - accuracy: 0.9705 - auc: 0.9976 - loss: 0.0943 - precision: 0.9726 - recall: 0.9677 - val_accuracy: 0.9084 - val_auc: 0.9881 - val_loss: 0.2509 - val_precision: 0.9185 - val_recall: 0.8947 - learning_rate: 1.0000e-04
Epoch 7/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 159s 860ms/step - accuracy: 0.9730 - auc: 0.9985 - loss: 0.0825 - precision: 0.9762 - recall: 0.9700 - val_accuracy: 0.7389 - val_auc: 0.9128 - val_loss: 1.0063 - val_precision: 0.7450 - val_recall: 0.7359 - learning_rate: 1.0000e-04
Epoch 8/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 0s 743ms/step - accuracy: 0.9702 - auc: 0.9984 - loss: 0.0866 - precision: 0.9726 - recall: 0.9681
Epoch 8: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
179/179 ━━━━━━━━━━━━━━━━━━━━ 160s 865ms/step - accuracy: 0.9702 - auc: 0.9984 - loss: 0.0866 - precision: 0.9726 - recall: 0.9681 - val_accuracy: 0.6550 - val_auc: 0.8603 - val_loss: 1.5468 - val_precision: 0.6630 - val_recall: 0.6458 - learning_rate: 1.0000e-04
Epoch 9/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 160s 863ms/step - accuracy: 0.9847 - auc: 0.9985 - loss: 0.0590 - precision: 0.9857 - recall: 0.9840 - val_accuracy: 0.9664 - val_auc: 0.9981 - val_loss: 0.1003 - val_precision: 0.9694 - val_recall: 0.9664 - learning_rate: 5.0000e-05
Epoch 10/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 159s 861ms/step - accuracy: 0.9880 - auc: 0.9996 - loss: 0.0355 - precision: 0.9892 - recall: 0.9873 - val_accuracy: 0.9802 - val_auc: 0.9991 - val_loss: 0.0632 - val_precision: 0.9802 - val_recall: 0.9802 - learning_rate: 5.0000e-05
Epoch 11/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 156s 841ms/step - accuracy: 0.9888 - auc: 0.9998 - loss: 0.0326 - precision: 0.9900 - recall: 0.9888 - val_accuracy: 0.9634 - val_auc: 0.9985 - val_loss: 0.1026 - val_precision: 0.9663 - val_recall: 0.9618 - learning_rate: 5.0000e-05
Epoch 12/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 157s 846ms/step - accuracy: 0.9902 - auc: 0.9995 - loss: 0.0341 - precision: 0.9905 - recall: 0.9901 - val_accuracy: 0.9053 - val_auc: 0.9815 - val_loss: 0.3584 - val_precision: 0.9133 - val_recall: 0.9008 - learning_rate: 5.0000e-05
Epoch 13/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 0s 729ms/step - accuracy: 0.9903 - auc: 0.9998 - loss: 0.0283 - precision: 0.9915 - recall: 0.9897
Epoch 13: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
179/179 ━━━━━━━━━━━━━━━━━━━━ 157s 849ms/step - accuracy: 0.9903 - auc: 0.9998 - loss: 0.0283 - precision: 0.9915 - recall: 0.9897 - val_accuracy: 0.8076 - val_auc: 0.9682 - val_loss: 0.5081 - val_precision: 0.8262 - val_recall: 0.7985 - learning_rate: 5.0000e-05
Epoch 14/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 159s 859ms/step - accuracy: 0.9944 - auc: 0.9998 - loss: 0.0217 - precision: 0.9945 - recall: 0.9938 - val_accuracy: 0.9710 - val_auc: 0.9991 - val_loss: 0.0725 - val_precision: 0.9725 - val_recall: 0.9710 - learning_rate: 2.5000e-05
Epoch 15/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 159s 860ms/step - accuracy: 0.9973 - auc: 1.0000 - loss: 0.0118 - precision: 0.9973 - recall: 0.9973 - val_accuracy: 0.9954 - val_auc: 1.0000 - val_loss: 0.0143 - val_precision: 0.9954 - val_recall: 0.9954 - learning_rate: 2.5000e-05
Epoch 16/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 157s 848ms/step - accuracy: 0.9963 - auc: 1.0000 - loss: 0.0092 - precision: 0.9965 - recall: 0.9963 - val_accuracy: 0.9786 - val_auc: 0.9995 - val_loss: 0.0536 - val_precision: 0.9786 - val_recall: 0.9786 - learning_rate: 2.5000e-05
Epoch 17/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 156s 840ms/step - accuracy: 0.9966 - auc: 0.9998 - loss: 0.0094 - precision: 0.9966 - recall: 0.9961 - val_accuracy: 0.9328 - val_auc: 0.9874 - val_loss: 0.2342 - val_precision: 0.9328 - val_recall: 0.9328 - learning_rate: 2.5000e-05
Epoch 18/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 0s 727ms/step - accuracy: 0.9965 - auc: 1.0000 - loss: 0.0107 - precision: 0.9965 - recall: 0.9965
Epoch 18: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.
179/179 ━━━━━━━━━━━━━━━━━━━━ 157s 848ms/step - accuracy: 0.9965 - auc: 1.0000 - loss: 0.0107 - precision: 0.9965 - recall: 0.9965 - val_accuracy: 0.9542 - val_auc: 0.9943 - val_loss: 0.1583 - val_precision: 0.9541 - val_recall: 0.9527 - learning_rate: 2.5000e-05
Epoch 19/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 158s 852ms/step - accuracy: 0.9980 - auc: 0.9999 - loss: 0.0109 - precision: 0.9980 - recall: 0.9980 - val_accuracy: 0.9939 - val_auc: 0.9999 - val_loss: 0.0215 - val_precision: 0.9939 - val_recall: 0.9939 - learning_rate: 1.2500e-05
Epoch 20/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 158s 855ms/step - accuracy: 0.9983 - auc: 0.9999 - loss: 0.0077 - precision: 0.9983 - recall: 0.9983 - val_accuracy: 0.9908 - val_auc: 0.9999 - val_loss: 0.0209 - val_precision: 0.9908 - val_recall: 0.9908 - learning_rate: 1.2500e-05
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
Train Loss: 0.0085
Train Accuracy: 99.72%
--------------------
Validation Loss: 0.0151
Validation Accuracy: 99.54%
--------------------
Test Loss: 0.0389
Test Accuracy: 99.39%
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

           0       1.00      0.99      0.99       150
           1       0.97      1.00      0.99       153
           2       1.00      1.00      1.00       203
           3       1.00      0.99      0.99       150

    accuracy                           0.99       656
   macro avg       0.99      0.99      0.99       656
weighted avg       0.99      0.99      0.99       656
"""

## 5.2 Testing
def predict_with_tta(model, img_path, num_augmentations=5):
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img_array = np.asarray(resized_img)
    
    # Create augmented versions
    predictions = []
    aug = get_augmentation()
    
    # Original prediction
    base_pred = model.predict(np.expand_dims(img_array, 0)/255.0)
    predictions.append(base_pred)
    
    # Augmented predictions
    for _ in range(num_augmentations):
        aug_img = aug.random_transform(img_array)
        aug_pred = model.predict(np.expand_dims(aug_img, 0)/255.0)
        predictions.append(aug_pred)
    
    # Average predictions
    return np.mean(predictions, axis=0)

def predict(img_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    label = list(class_dict.keys())
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    
    # Use TTA for prediction
    predictions = predict_with_tta(model, img_path)
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

predict('/kaggle/input/Testing/meningioma/Te-meTr_0000.jpg')
# it predicted "meningioma" with 1.00 probability
predict('/kaggle/input/Testing/meningioma/Te-me_0010.jpg')
# it predicted "meningioma" with 1.00 probability


predict('/kaggle/input/Testing/glioma/Te-glTr_0007.jpg')
# it predicted "glioma" with 1.00 probability
predict('/kaggle/input/Testing/glioma/Te-gl_0017.jpg')
# it predicted "glioma" with 1.00 probability


predict('/kaggle/input/Testing/notumor/Te-noTr_0001.jpg')
# it predicted "notumor" with 1.00 probability
predict('/kaggle/input/Testing/notumor/Te-no_0011.jpg')
# it predicted "notumor" with 0.99 probability


predict('/kaggle/input/Testing/pituitary/Te-piTr_0001.jpg')
# it predicted "pituitary" with 1.00 probability
predict('/kaggle/input/Testing/pituitary/Te-pi_0011.jpg')
# it predicted "pituitary" with 1.00 probability

