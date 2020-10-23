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
import os

#deep learning hyperparameters
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#Load Our Training Data

#Path of data set folder
DIRECTORY = r"dataset"
#Categories inside data set folder
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("Please wait images are loading...")

#data containes image array
data = []
#all those corresponding label to indicate that image contain mask or not
labels = []

#Loop Over the image path
for category in CATEGORIES:
	#Join the path and category to navigate to location
    path = os.path.join(DIRECTORY, category)
    #Read all the image in that path
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)

    	#Load the input image and preprocess it
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	#Update the Data and Label Lists, Respectively
    	data.append(image)
    	labels.append(category)


# perform one-hot encoding on the labels
# One hot encoding is a process by which categorical variables are converted into a 
# form that could be provided to ML algorithms to do a better job in prediction.
#Yeh label lagadega mask wala images array ko zero and and without mask wala images array ko one
#array([[0., 1.],
#       [0., 1.],
 #      [0., 1.],
  #     [1., 0.],
   #    [1., 0.],
    #   [0., 1.],
     #  [0., 1.]], dtype=float32)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Converting our training data in NumPy array format
data = np.array(data, dtype="float32")
labels = np.array(labels)

#the scikit-learn library provides a train_test_split to split the data in training set and testing set.
#20% testing and 80% training
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
# Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly
# modified copies of already existing data or newly created synthetic data from existing data. 
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
# MobileNetV2 for fine-tuning
# Fine-tuning, in general, means making small adjustments to a process to achieve the desired output or performance.
# Load MobileNet with pre-trained ImageNet weights, leaving off head of network
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
# compile our model with the Adam optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Face mask training is launched
# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# make predictions on the test set, grabbing the highest probability class label indices. 
# Then, we print a classification report in the terminal for inspection.
# show a nicely formatted classification reportvo
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serializes our face mask classification model to disk.
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")