from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow as tf
import pandas as pd
from generator import ImageSequence
from callback import MultipleClassAUROC

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

n_epoch = 100
batch_size = 32

train_sequence = ImageSequence(
    dataset_csv_file="",
    class_names=["Airspace Opacity", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"],
    source_image_dir="",
    batch_size=batch_size,
    target_size=(224, 224)
)
val_sequence = ImageSequence(
    dataset_csv_file="",
    class_names=["Airspace Opacity", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"],
    source_image_dir="",
    batch_size=batch_size,
    target_size=(224, 224),
    shuffle_on_epoch_end=False
)

# model
input = Input(shape=(224, 224, 3))
model = DenseNet121(include_top=False, weights="imagenet", input_tensor=input, pooling="avg")

x = model.output
predictions = Dense(13, activation="sigmoid", name="output")(x)
model = Model(input=input, output=predictions)

model.compile(optimizer=Adam(lr=0.001),
              loss="binary_crossentropy",
              metrics=["mse", "acc"])


# train
filepath = ""
checkpoint = ModelCheckpoint(
    filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
tb = TensorBoard(log_dir="tb/", histogram_freq=0,
                 write_graph=True, write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=1,
                              verbose=1, mode="min", min_lr=1e-8)
auroc = MultipleClassAUROC(
            sequence=val_sequence,
            class_names=["Airspace Opacity", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"],
            weights_path="models/weights.densenet121_2019_5_5_nyu.hdf5"
        )
history_ft = model.fit_generator(
    train_sequence,
    steps_per_epoch=len(train_sequence),
    epochs=n_epoch,
    validation_data=val_sequence,
    validation_steps=len(val_sequence),
    callbacks=[checkpoint, tb, reduce_lr, auroc]
)
