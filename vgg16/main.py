# vgg16/main.py
# v6

# ------------------------------
# 1. Import needed libraries
# ------------------------------

# ---------- Basic Python imports ----------
import os
import time
import warnings
# ---------- Image Processing ----------
from PIL import Image
# ---------- Data Analysis & Visualization ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ---------- Machine Learning ----------
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
# ---------- Deep Learning & TensorFlow ----------
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Conv2D,
    Multiply,
    GlobalAveragePooling2D,
    Reshape,
    Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# ---------- Settings ----------
warnings.filterwarnings("ignore")


# GPU/CPU detection
print("Checking available devices...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        strategy = tf.distribute.MirroredStrategy()
        print(f"Running on {len(strategy.extended.worker_devices)} GPU(s)")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        strategy = tf.distribute.get_strategy()
        print("Running on CPU")
else:
    strategy = tf.distribute.get_strategy()
    print("No GPU detected. Running on CPU")

print("REPLICAS: ", strategy.num_replicas_in_sync)


# ------------------------------
# 2. Preprocessing
# ------------------------------

# 2.1 Load Data
# ------------------------------

def create_dataset_df(path):
    """
    Create a DataFrame containing image paths and their corresponding classes.

    Args:
        path (str): Path to the dataset directory

    Returns:
        pd.DataFrame: DataFrame with columns 'Class Path' and 'Class'
    """
    classes, class_paths = zip(*[(label, os.path.join(path, label, image))
                                 for label in os.listdir(path) if os.path.isdir(os.path.join(path, label))
                                 for image in os.listdir(os.path.join(path, label))])

    return pd.DataFrame({'Class Path': class_paths, 'Class': classes})

# /kaggle/input/


tr_df = create_dataset_df('/brain-tumor-mri-dataset/Training')
tr_df

ts_df = create_dataset_df('/brain-tumor-mri-dataset/Testing')
ts_df


# Count of images in each class in train data
plt.figure(figsize=(15, 7))
ax = sns.countplot(data=tr_df, y=tr_df['Class'])

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


# 2.2 Split data into train, test, valid
# ------------------------------

valid_df, ts_df = train_test_split(
    ts_df, train_size=0.5, random_state=20, stratify=ts_df['Class'])

valid_df


# 2.3 Data preprocessing
# ------------------------------

BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = (224, 224)


def prepare_data(tr_df, ts_df):
    def process_image(file_path):
        img = Image.open(file_path)
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMAGE_SIZE)
        return np.array(img)

    # Convert DataFrame to numpy arrays
    X = np.array([process_image(f)
                 for f in tr_df['Class Path']], dtype=np.float32) / 255.0
    y = pd.get_dummies(tr_df['Class']).values

    X_test = np.array([process_image(f)
                      for f in ts_df['Class Path']], dtype=np.float32) / 255.0
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


# 2.4 Getting samples from data
# ------------------------------

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


# ------------------------------
# 3. Building Deep Learning Model
# ------------------------------

class SAM(Model):
    def __init__(self, filters):
        super(SAM, self).__init__()
        self.filters = filters
        # Three sequential 3x3 convs as specified
        self.conv1 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        # Dimension reduction conv
        self.conv4 = Conv2D(self.filters // 4, 1,
                            activation='relu', kernel_initializer='he_normal')
        # Attention branch convs
        self.W1 = Conv2D(1, 1, activation='sigmoid',
                         kernel_initializer='he_normal')
        self.W2 = Conv2D(1, 1, activation='sigmoid',
                         kernel_initializer='he_normal')

    def call(self, inputs):
        # Sequential convolutions
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        # Dimension reduction
        out2 = self.conv4(inputs)

        # 2x2 max pooling branch
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(out2)
        # Bilinear upsampling to original size
        upsample1 = tf.image.resize(pool1, size=tf.shape(out2)[
                                    1:3], method='bilinear')
        # Apply 1x1 conv with sigmoid
        attention1 = self.W1(upsample1)

        # 4x4 max pooling branch
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4))(out2)
        # Bilinear upsampling to original size
        upsample2 = tf.image.resize(pool2, size=tf.shape(out2)[
                                    1:3], method='bilinear')
        # Apply 1x1 conv with sigmoid
        attention2 = self.W2(upsample2)

        # Sum the two attention maps
        attention_sum = attention1 + attention2

        # Apply attention to features via element-wise multiplication
        attended_features = Multiply()([out1, attention_sum])

        # Add to dimension-reduced input (residual connection)
        y = attended_features + out2
        return y


class CAM(Model):
    def __init__(self, filters, reduction_ratio=16):
        super(CAM, self).__init__()
        self.filters = filters
        # Conv block to process input features
        self.conv1 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(self.filters // 4, 3, activation='relu',
                            padding='same', kernel_initializer='he_normal')
        # Dimension reduction conv
        self.conv4 = Conv2D(self.filters // 4, 1,
                            activation='relu', kernel_initializer='he_normal')
        # Squeeze-and-Excitation components
        self.gpool = GlobalAveragePooling2D()
        self.fc1 = Dense(self.filters // (4 * reduction_ratio),
                         activation='relu', use_bias=False)
        self.fc2 = Dense(self.filters // 4,
                         activation='sigmoid', use_bias=False)

    def call(self, inputs):
        # Process input through conv block
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        # Dimension reduction
        out2 = self.conv4(inputs)

        # Squeeze-and-Excitation: squeeze spatial dimensions
        channel_attention = self.gpool(out2)
        # Dimension reduction in channel-wise fully connected layer
        channel_attention = self.fc1(channel_attention)
        # Dimension increase with sigmoid activation
        channel_attention = self.fc2(channel_attention)
        # Reshape to proper dimensions for broadcasting
        channel_attention = Reshape(
            (1, 1, self.filters // 4))(channel_attention)

        # Apply channel attention via element-wise multiplication
        recalibrated = Multiply()([out1, channel_attention])

        # Add residual connection with dimension-reduced input
        y = recalibrated + out2
        return y


class SynergyModule(Model):
    def __init__(self, filters):
        super(SynergyModule, self).__init__()
        self.filters = filters
        # Trainable scaling parameters
        self.alpha = tf.Variable(
            0.5, trainable=True, dtype=tf.float32, name="alpha")
        self.beta = tf.Variable(0.5, trainable=True,
                                dtype=tf.float32, name="beta")
        # Integration components
        self.conv = Conv2D(filters, 3, padding='same',
                           kernel_initializer='he_normal')
        self.bn = BatchNormalization()

    def call(self, inputs):
        # Unpack inputs (spatial and channel attention outputs)
        spatial_features, channel_features = inputs

        # Scale each pathway with trainable parameters
        scaled_spatial = tf.multiply(spatial_features, self.alpha)
        scaled_channel = tf.multiply(channel_features, self.beta)

        # Sum the scaled features
        combined = scaled_spatial + scaled_channel

        # Apply convolution and batch normalization
        output = self.conv(combined)
        output = self.bn(output)

        return output


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
def AS_Net(encoder='vgg16', input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), fine_tune_at=None):
    inputs = Input(input_size)
    print(f'CURRENT ENCODER: {encoder}')

    if encoder == 'vgg16':
        # Load VGG16 with ImageNet weights
        ENCODER = VGG16(weights='imagenet', include_top=False,
                        input_shape=input_size)

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
        raise ValueError(
            "Unsupported encoder type. Only 'vgg16' is supported in this case.")

    # Get the output layers dynamically
    output_layers = [ENCODER.get_layer(index=i).output for i in layer_indices]
    encoder_outputs = [Model(inputs=ENCODER.inputs, outputs=layer)(inputs)
                       for layer in output_layers]

    # Get the final encoder output that will be processed by the attention paths
    final_encoder_output = encoder_outputs[-1]
    filters = final_encoder_output.shape[-1]

    # Create two parallel attention paths
    # Spatial Attention Path
    SAM_output = SAM(filters=filters)(final_encoder_output)

    # Channel Attention Path
    CAM_output = CAM(filters=filters)(final_encoder_output)

    # Apply Synergy Module to combine attention outputs
    synergy_output = SynergyModule(filters=filters)([SAM_output, CAM_output])

    # Simplify the final layers
    final_layers = Sequential([
        Conv2D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    output = final_layers(synergy_output)

    model = Model(inputs=inputs, outputs=output)
    return model


# Create and compile the model
with strategy.scope():
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


# ------------------------------
# 4. Training
# ------------------------------


start_time = time.time()
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
179/179 ━━━━━━━━━━━━━━━━━━━━ 134s 658ms/step - accuracy: 0.7955 - auc: 0.9416 - loss: 0.5785 - precision: 0.8809 - recall: 0.6757 - val_accuracy: 0.8931 - val_auc: 0.9791 - val_loss: 0.3290 - val_precision: 0.9057 - val_recall: 0.8794 - learning_rate: 1.0000e-04
Epoch 2/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 91s 492ms/step - accuracy: 0.9191 - auc: 0.9889 - loss: 0.2501 - precision: 0.9299 - recall: 0.9025 - val_accuracy: 0.6748 - val_auc: 0.8964 - val_loss: 1.0807 - val_precision: 0.6822 - val_recall: 0.6718 - learning_rate: 1.0000e-04
Epoch 3/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 92s 498ms/step - accuracy: 0.9433 - auc: 0.9930 - loss: 0.1862 - precision: 0.9515 - recall: 0.9310 - val_accuracy: 0.8168 - val_auc: 0.9605 - val_loss: 0.4962 - val_precision: 0.8299 - val_recall: 0.7969 - learning_rate: 1.0000e-04
Epoch 4/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 92s 499ms/step - accuracy: 0.9506 - auc: 0.9955 - loss: 0.1499 - precision: 0.9581 - recall: 0.9460 - val_accuracy: 0.9206 - val_auc: 0.9920 - val_loss: 0.2127 - val_precision: 0.9314 - val_recall: 0.9115 - learning_rate: 1.0000e-04
Epoch 5/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 93s 499ms/step - accuracy: 0.9551 - auc: 0.9977 - loss: 0.1154 - precision: 0.9593 - recall: 0.9498 - val_accuracy: 0.9313 - val_auc: 0.9954 - val_loss: 0.1712 - val_precision: 0.9368 - val_recall: 0.9282 - learning_rate: 1.0000e-04
Epoch 6/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 91s 493ms/step - accuracy: 0.9691 - auc: 0.9975 - loss: 0.0959 - precision: 0.9728 - recall: 0.9645 - val_accuracy: 0.9160 - val_auc: 0.9881 - val_loss: 0.2433 - val_precision: 0.9297 - val_recall: 0.9084 - learning_rate: 1.0000e-04
Epoch 7/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 91s 492ms/step - accuracy: 0.9691 - auc: 0.9975 - loss: 0.0966 - precision: 0.9719 - recall: 0.9655 - val_accuracy: 0.6397 - val_auc: 0.8406 - val_loss: 1.7129 - val_precision: 0.6400 - val_recall: 0.6351 - learning_rate: 1.0000e-04
Epoch 8/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 0s 421ms/step - accuracy: 0.9780 - auc: 0.9985 - loss: 0.0747 - precision: 0.9807 - recall: 0.9754
Epoch 8: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
179/179 ━━━━━━━━━━━━━━━━━━━━ 91s 492ms/step - accuracy: 0.9780 - auc: 0.9985 - loss: 0.0747 - precision: 0.9807 - recall: 0.9754 - val_accuracy: 0.8718 - val_auc: 0.9712 - val_loss: 0.4050 - val_precision: 0.8834 - val_recall: 0.8672 - learning_rate: 1.0000e-04
Epoch 9/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 92s 500ms/step - accuracy: 0.9841 - auc: 0.9991 - loss: 0.0527 - precision: 0.9848 - recall: 0.9811 - val_accuracy: 0.9756 - val_auc: 0.9981 - val_loss: 0.0722 - val_precision: 0.9785 - val_recall: 0.9710 - learning_rate: 5.0000e-05
Epoch 10/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 94s 507ms/step - accuracy: 0.9884 - auc: 0.9985 - loss: 0.0466 - precision: 0.9894 - recall: 0.9867 - val_accuracy: 0.9847 - val_auc: 0.9997 - val_loss: 0.0438 - val_precision: 0.9877 - val_recall: 0.9847 - learning_rate: 5.0000e-05
Epoch 11/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 93s 500ms/step - accuracy: 0.9938 - auc: 0.9998 - loss: 0.0238 - precision: 0.9945 - recall: 0.9929 - val_accuracy: 0.9817 - val_auc: 0.9997 - val_loss: 0.0404 - val_precision: 0.9832 - val_recall: 0.9817 - learning_rate: 5.0000e-05
Epoch 12/35
...
Epoch 20/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 96s 517ms/step - accuracy: 0.9987 - auc: 1.0000 - loss: 0.0072 - precision: 0.9988 - recall: 0.9987 - val_accuracy: 0.9939 - val_auc: 1.0000 - val_loss: 0.0116 - val_precision: 0.9939 - val_recall: 0.9939 - learning_rate: 1.2500e-05
Epoch 21/35
179/179 ━━━━━━━━━━━━━━━━━━━━ 96s 517ms/step - accuracy: 0.9983 - auc: 0.9998 - loss: 0.0069 - precision: 0.9983 - recall: 0.9981 - val_accuracy: 0.9954 - val_auc: 1.0000 - val_loss: 0.0156 - val_precision: 0.9954 - val_recall: 0.9954 - learning_rate: 1.2500e-05
"""


end_time = time.time()
training_duration = end_time - start_time

# Print duration in hours, minutes, and seconds
hours = int(training_duration // 3600)
minutes = int((training_duration % 3600) // 60)
seconds = int(training_duration % 60)
print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")


"""
Total training time: 00:35:41
"""


hist.history.keys()


# 4.1 Visualize model performance
# ------------------------------

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
plt.scatter(index_recall + 1, recall_highest,
            s=150, c='blue', label=recall_label)
plt.title('Recall and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.show()


# ------------------------------
# 5. Testing and Evaluation
# ------------------------------

# 5.1 Evaluate
# ------------------------------

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
Train Loss: 0.0056
Train Accuracy: 99.88%
--------------------
Validation Loss: 0.0215
Validation Accuracy: 99.54%
--------------------
Test Loss: 0.0305
Test Accuracy: 99.24%
"""


preds = model.predict(ts_gen)
y_pred = np.argmax(preds, axis=1)


cm = confusion_matrix(ts_gen.classes, y_pred)
labels = list(class_dict.keys())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('Truth Label')
plt.show()


clr = classification_report(ts_gen.classes, y_pred)
print(clr)


"""
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       150
           1       0.99      0.99      0.99       153
           2       1.00      1.00      1.00       203
           3       0.99      0.99      0.99       150

    accuracy                           0.99       656
   macro avg       0.99      0.99      0.99       656
weighted avg       0.99      0.99      0.99       656
"""


# ------------------------------
# 5.2 Testing
# ------------------------------

def predict_with_tta(model, img_path, num_augmentations=5):
    img = Image.open(img_path)
    resized_img = img.resize(IMAGE_SIZE)
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
    label = list(class_dict.keys())
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize(IMAGE_SIZE)

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
    ax.bar_label(bars, fmt='%.2f')
    plt.show()


predict('/brain-tumor-mri-dataset/Testing/meningioma/Te-meTr_0000.jpg')
# 0.98 meningioma
predict('/brain-tumor-mri-dataset/Testing/glioma/Te-glTr_0007.jpg')
# 1.0 glioma
predict('/brain-tumor-mri-dataset/Testing/notumor/Te-noTr_0001.jpg')
# 1.0 notumor
predict('/brain-tumor-mri-dataset/Testing/pituitary/Te-piTr_0001.jpg')
# 1.0 pituitary
