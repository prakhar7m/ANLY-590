# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
parser= argparse.ArgumentParser()

parser.add_argument("-d", "--data", required=True)
parser.add_argument("-m", "--model", type=str,
	default="mask_detector.model")
args = vars(parser.parse_args())


# get the images in our data set
print("loading images...")
im_list = list(paths.list_images(args["dataset"]))

#create empty list variables
data = []
labels = []

# loop over the image paths
for i in im_list:
	# extract the class label from the filename
	label = i.split(os.path.sep)[-2]
	# load the input image (224x224) and preprocess it
	image = load_img(i, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# hot encoding the labels
labels = LabelBinarizer().fit_transform(labels)
labels = to_categorical(labels)

# *0:20 train test split
(train_X, test_X, train_Y, test_Y) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=50)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Load the mobilenetv2

baseModel= MobileNetV2(input_shape=None,
                       alpha=1.0,
    				   include_top=False,
    				   weights='imagenet',
    				   input_tensor=Input(shape=(224, 224, 3)),
   					   pooling=None,
    				   classes=1000,
    				   classifier_activation='softmax')


model_2 = baseModel.output
model_2 = AveragePooling2D(pool_size=(7, 7))(model_2)
model_2 = Flatten(name="flatten")(model_2)
model_2= Dense(128, activation="relu")(model_2)
model_2 = Dropout(0.5)(model_2)
model_2 = Dense(2, activation="softmax")(model_2)

# pmodel_2 to be placed over base model
model = Model(inputs=baseModel.input, outputs=model_2)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model

# learning rate, epoch and batchsize initialization
INIT_LR = .0001
EPOCHS = 10
BS = 32

print("model is being compiled...")
optimise = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy",
              optimizer=optimise,
              metrics=["accuracy"])



#for graphs
"""History=model.fit(aug.flow(train_X, train_Y, batch_size=32),
    x=None,
    y=None,
    batch_size=None,
    epochs=10,
    verbose='auto',
    callbacks=None,
    validation_split=0.0,
    validation_data=(test_X, test_Y),
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=len(train_X) // 32,
    validation_steps=len(test_X) // 32,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
"""
# make predictions on the testing set
print("Evaluating..")
pred = model.predict(test_X, batch_size=BS)


# serialize the model to disk
print("Save model..")
model.save(args["model"], save_format="h5")

