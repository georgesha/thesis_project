from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow as tf
import pandas as pd
from generator_2input import ImageSequence
from callback import MultipleClassAUROC

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

n_epoch = 50
batch_size = 16

train_sequence = ImageSequence(
    dataset_csv_file="",
    class_names=["Airspace Opacity", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                "Fracture", "Lung Lesion", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"],
    source_image_dir="",
    batch_size=batch_size,
    target_size=(224, 224)
)
val_sequence = ImageSequence(
    dataset_csv_file="",
    class_names=["Airspace Opacity", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                "Fracture", "Lung Lesion", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"],
    source_image_dir="",
    batch_size=batch_size,
    target_size=(224, 224),
    shuffle_on_epoch_end=False
)

input_pa = Input(shape=(224, 224, 3))
input_lat = Input(shape=(224, 224, 3))

model_pa = DenseNet121(include_top=False, weights="imagenet",
                    input_tensor=input_pa, pooling="avg")
model_lat = DenseNet121(include_top=False, weights="imagenet",
                    input_tensor=input_lat, pooling="avg")

for i, layer in enumerate(model_lat.layers):
    layer.name = model_pa.layers[i].name + "_lat"

merge = concatenate([model_pa.output, model_lat.output])

predictions = Dense(13, activation="sigmoid", name="output")(merge)
model = Model(input=[input_pa, input_lat], output=predictions)

model.compile(optimizer=Adam(lr=0.001),
              loss="binary_crossentropy",
              metrics=["mse", "acc"])

# train
filepath = "models/weights.densenet_nyu_2input.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
tb = TensorBoard(log_dir="tb/", histogram_freq=0,
                 write_graph=True, write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=1,
                              verbose=1, mode="min", min_lr=1e-5)
auroc = MultipleClassAUROC(
    sequence=val_sequence,
    class_names=["Airspace Opacity", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                "Fracture", "Lung Lesion", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"],
    weights_path="models/weights.densenet_nyu_2input.hdf5"
)
history_ft = model.fit_generator(
    train_sequence,
    steps_per_epoch=len(train_sequence),
    epochs=n_epoch,
    validation_data=val_sequence,
    validation_steps=len(val_sequence),
    callbacks=[checkpoint, tb, reduce_lr, auroc]
)
