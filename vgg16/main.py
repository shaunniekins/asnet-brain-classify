# v3.2.3

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
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l1_l2
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
    # Convert DataFrame to numpy arrays
    X = np.array([np.array(Image.open(f).resize(IMAGE_SIZE))/255.0 
                  for f in tr_df['Class Path']])
    y = pd.get_dummies(tr_df['Class']).values
    
    X_test = np.array([np.array(Image.open(f).resize(IMAGE_SIZE))/255.0 
                       for f in ts_df['Class Path']])
    y_test = pd.get_dummies(ts_df['Class']).values
    
    return X, y, X_test, y_test

# Modify get_augmentation to include more robust augmentations
def get_augmentation():
    return ImageDataGenerator(
        rescale=1/255,
        brightness_range=(0.9, 1.1),    # Reduced range for more stable training
        rotation_range=15,              # Slight increase in rotation
        width_shift_range=0.15,         # Increased shift range
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,             # Added vertical flip
        fill_mode='reflect',
        preprocessing_function=lambda x: x + np.random.normal(0, 0.02, x.shape)  # Slight noise
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


# Modified AS_Net with VGG16 encoder
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

    # Final classification layers (added more layers and dropout)
    final_layers = Sequential([
        Conv2D(256, 3, activation='relu', padding='same',
               kernel_regularizer=l1_l2(reg_factor, reg_factor)),
        BatchNormalization(),
        Conv2D(128, 3, activation='relu', padding='same',
               kernel_regularizer=l1_l2(reg_factor, reg_factor)),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(reg_factor, reg_factor)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(reg_factor, reg_factor)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l1_l2(reg_factor, reg_factor)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(4, activation='softmax', kernel_initializer='he_normal')
    ])

    output = final_layers(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

# Create and compile the model
with tpu_strategy.scope():
    # Create and compile the model
    model = AS_Net(encoder='vgg16', fine_tune_at=12, reg_factor=0.0005)
    
    # Improved learning rate scheduler with warmup
    initial_learning_rate = 1e-4
    warmup_epochs = 5
    decay_epochs = 15
    total_steps = tr_gen.samples * warmup_epochs // BATCH_SIZE
    decay_steps = tr_gen.samples * decay_epochs // BATCH_SIZE
    
    # Use CosineDecay instead of CosineDecaySchedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=1e-6,  # Minimum learning rate factor
        name='cosine_decay'
    )
    
    optimizer = Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    # Fixed metrics instantiation
    precision_metric = Precision(name='precision')
    recall_metric = Recall(name='recall')
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', precision_metric, recall_metric],
        steps_per_execution=32
    )

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)


# 4. Training
num_epochs = 35 #20

# Add TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=1, write_graph=True, profile_batch=0)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

# Add learning rate scheduler callback
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# Compute class weights
def get_class_weights(y):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(class_weights))

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(tr_gen.classes),
    y=tr_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))

# Modified training configuration
hist = model.fit(
    tr_gen,
    epochs=num_epochs,
    validation_data=valid_gen,
    shuffle=True,  # Enable shuffling
    class_weight=class_weight_dict,
    callbacks=[
        early_stopping,
        tensorboard_callback,
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    ]
)

"""
Epoch 1/35
179/179 [==============================] - 223s 1s/step - loss: 23.6039 - accuracy: 0.5835 - precision: 0.6253 - recall: 0.5217 - val_loss: 26.0893 - val_accuracy: 0.3084 - val_precision: 0.3084 - val_recall: 0.3084 - lr: 9.8913e-05
Epoch 2/35
179/179 [==============================] - 213s 1s/step - loss: 20.4491 - accuracy: 0.7752 - precision: 0.8000 - recall: 0.7442 - val_loss: 22.4465 - val_accuracy: 0.3084 - val_precision: 0.3084 - val_recall: 0.3084 - lr: 9.5676e-05
Epoch 3/35
179/179 [==============================] - 216s 1s/step - loss: 16.2019 - accuracy: 0.8377 - precision: 0.8535 - recall: 0.8193 - val_loss: 14.8666 - val_accuracy: 0.6565 - val_precision: 0.7342 - val_recall: 0.6366 - lr: 9.0430e-05
Epoch 4/35
179/179 [==============================] - 213s 1s/step - loss: 13.1291 - accuracy: 0.8592 - precision: 0.8731 - recall: 0.8470 - val_loss: 12.9865 - val_accuracy: 0.6504 - val_precision: 0.6822 - val_recall: 0.6260 - lr: 8.3407e-05
Epoch 5/35
179/179 [==============================] - 211s 1s/step - loss: 11.4908 - accuracy: 0.8866 - precision: 0.8990 - recall: 0.8789 - val_loss: 11.0077 - val_accuracy: 0.8443 - val_precision: 0.8676 - val_recall: 0.8107 - lr: 7.4915e-05
Epoch 6/35
179/179 [==============================] - 214s 1s/step - loss: 10.4592 - accuracy: 0.8992 - precision: 0.9081 - recall: 0.8897 - val_loss: 10.2751 - val_accuracy: 0.8046 - val_precision: 0.8272 - val_recall: 0.7893 - lr: 6.5328e-05
Epoch 7/35
179/179 [==============================] - 212s 1s/step - loss: 9.7005 - accuracy: 0.9175 - precision: 0.9234 - recall: 0.9081 - val_loss: 9.6819 - val_accuracy: 0.8321 - val_precision: 0.8404 - val_recall: 0.8198 - lr: 5.5067e-05
Epoch 8/35
179/179 [==============================] - 215s 1s/step - loss: 9.1452 - accuracy: 0.9266 - precision: 0.9340 - recall: 0.9212 - val_loss: 9.4258 - val_accuracy: 0.7740 - val_precision: 0.7786 - val_recall: 0.7573 - lr: 4.4583e-05
Epoch 9/35
179/179 [==============================] - 213s 1s/step - loss: 8.7467 - accuracy: 0.9331 - precision: 0.9396 - recall: 0.9266 - val_loss: 8.9072 - val_accuracy: 0.8427 - val_precision: 0.8617 - val_recall: 0.8275 - lr: 3.4337e-05
Epoch 10/35
179/179 [==============================] - 211s 1s/step - loss: 8.4251 - accuracy: 0.9426 - precision: 0.9471 - recall: 0.9363 - val_loss: 9.0212 - val_accuracy: 0.7710 - val_precision: 0.7804 - val_recall: 0.7649 - lr: 2.4780e-05
Epoch 11/35
179/179 [==============================] - 214s 1s/step - loss: 8.1919 - accuracy: 0.9520 - precision: 0.9557 - recall: 0.9473 - val_loss: 8.2507 - val_accuracy: 0.9328 - val_precision: 0.9370 - val_recall: 0.9313 - lr: 1.6332e-05
Epoch 12/35
179/179 [==============================] - 213s 1s/step - loss: 8.0270 - accuracy: 0.9573 - precision: 0.9619 - recall: 0.9543 - val_loss: 8.0404 - val_accuracy: 0.9450 - val_precision: 0.9478 - val_recall: 0.9420 - lr: 9.3638e-06
Epoch 13/35
179/179 [==============================] - 214s 1s/step - loss: 7.9354 - accuracy: 0.9603 - precision: 0.9647 - recall: 0.9571 - val_loss: 8.0136 - val_accuracy: 0.9496 - val_precision: 0.9525 - val_recall: 0.9496 - lr: 4.1823e-06
Epoch 14/35
179/179 [==============================] - 211s 1s/step - loss: 7.8972 - accuracy: 0.9597 - precision: 0.9632 - recall: 0.9564 - val_loss: 7.9154 - val_accuracy: 0.9557 - val_precision: 0.9571 - val_recall: 0.9527 - lr: 1.0152e-06
Epoch 15/35
179/179 [==============================] - 211s 1s/step - loss: 7.8772 - accuracy: 0.9648 - precision: 0.9676 - recall: 0.9629 - val_loss: 7.9535 - val_accuracy: 0.9588 - val_precision: 0.9587 - val_recall: 0.9573 - lr: 1.0000e-10
Epoch 16/35
179/179 [==============================] - 208s 1s/step - loss: 7.8724 - accuracy: 0.9687 - precision: 0.9708 - recall: 0.9655 - val_loss: 7.9352 - val_accuracy: 0.9573 - val_precision: 0.9572 - val_recall: 0.9557 - lr: 1.0000e-10
Epoch 17/35
179/179 [==============================] - 215s 1s/step - loss: 7.8702 - accuracy: 0.9678 - precision: 0.9714 - recall: 0.9641 - val_loss: 7.9873 - val_accuracy: 0.9511 - val_precision: 0.9555 - val_recall: 0.9496 - lr: 1.0000e-10
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
Train Loss: 7.8299
Train Accuracy: 98.06%
--------------------
Validation Loss: 7.9119
Validation Accuracy: 96.03%
--------------------
Test Loss: 7.8770
Test Accuracy: 96.49%
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

           0       0.95      0.97      0.96       150
           1       0.97      0.89      0.93       153
           2       0.99      1.00      0.99       203
           3       0.94      0.99      0.96       150

    accuracy                           0.96       656
   macro avg       0.96      0.96      0.96       656
weighted avg       0.97      0.96      0.96       656
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

