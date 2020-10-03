
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
import csv

from tensorflow.keras import datasets, layers, models

def data_split(examples, labels, train_frac, random_state=5):
    '''
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset
    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2
    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(
        examples, labels, train_size=train_frac, random_state=random_state)

    return X_train, X_test, Y_train, Y_test


#READ the file
print("Started Reading File")
df = pd.read_csv(r"../../Documents/qmind/CampQMIND-BData/finalTrain.csv")
print("Finished Reading File")
df.info()

X_train, X_test, Y_train, Y_test, = data_split(examples=df.drop("label", axis=1),
                                                            labels=df["label"],train_frac=0.8)
#x=images, y=lables

print(f"Training size is {X_train.shape[0]}")
print(f"Test size is {X_test.shape[0]}")

train_labels = Y_train.to_numpy()
test_labels = Y_test.to_numpy()
train_images = X_train.to_numpy()
train_images_shaped = np.zeros((len(train_images), 28, 28, 1))
test_images = X_test.to_numpy()
test_images_shaped = np.zeros((len(test_images), 28, 28, 1))
for i in range(len(train_images)):
    train_images_shaped[i] = np.asarray(train_images[i]).reshape(28, 28, 1)
for i in range(len(test_images)):
    test_images_shaped[i] = np.asarray(test_images[i]).reshape(28, 28, 1)
#train_images_shaped = np.asarray(np.expand_dims(train_images_shaped, axis=3))
#test_images_shaped = np.asarray(np.expand_dims(train_images_shaped, axis=3))



for i in range(len(test_images_shaped)):
    test_images_shaped[i] = np.asarray(test_images_shaped[i])

for i in range(len(train_images_shaped)):
    train_images_shaped[i] = np.asarray(train_images_shaped[i])

train_images_shaped, test_images_shaped = train_images_shaped / 255.0, test_images_shaped / 255.0
print("Shape: " + str(np.shape(train_images_shaped)))

plot.figure(figsize=(10,10))
for i in range(25):
    plot.subplot(5,5,i+1)
    plot.xticks([])
    plot.yticks([])
    plot.grid(False)
    plot.imshow(train_images_shaped[i], cmap=plot.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plot.xlabel(train_labels[i])
plot.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.asarray(train_images_shaped), np.asarray(train_labels), epochs=10,
                    validation_data=(np.asarray(test_images_shaped),np.asarray(test_labels)))

plot.plot(history.history['accuracy'], label='accuracy')
plot.plot(history.history['val_accuracy'], label = 'val_accuracy')
plot.xlabel('Epoch')
plot.ylabel('Accuracy')
plot.ylim([0.5, 1])
plot.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

df_predict = pd.read_csv(r"../../Documents/qmind/CampQMIND-BData/finalPredict.csv")
predict_images = df_predict.to_numpy()
predict_images_shaped = np.zeros((len(predict_images), 28, 28))
for i in range(len(predict_images)):
    predict_images_shaped[i] = np.asarray(predict_images[i]).reshape(28, 28)
predict_images_shaped = np.expand_dims(train_images_shaped, axis=3)

prediction = model.predict_classes(predict_images_shaped)

for i in prediction:
    print(i)

with open("./prediction", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(prediction)

model.save("./model")

