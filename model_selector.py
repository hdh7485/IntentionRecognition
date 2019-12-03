from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import cifar10

from keras.applications import xception
from keras.applications import inception_resnet_v2
from keras.applications import resnet_v2
from keras.applications import mobilenet_v2
from keras.applications import densenet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

img_rows = 128
img_cols = 128

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

resize_x_train = []
for img in x_train:
    resize_x_train.append(cv2.resize(img, (img_rows, img_cols)))
x_train = np.array(resize_x_train)

model = models.Sequential()
image_size = 612

name = 'inception_resnet_v2'
model_dict = { 'xception': xception.Xception(weights='imagenet', include_top=False, input_shape=(img_cols, img_rows, 3)),
        'inception_resnet_v2': inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_cols, img_rows, 3)),
        'resnet_v2': resnet_v2.ResNet101V2(weights='imagenet', include_top=False),
        'mobilenet_v2': mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False),
        'densenet': densenet.DenseNet201(weights='imagenet', include_top=False)
        }
name_dict = { 'xception': xception,
        'inception_resnet_v2': inception_resnet_v2,
        'resnet_v2': resnet_v2,
        'mobilenet_v2': mobilenet_v2,
        'densenet': densenet
        }
pretrained_model = model_dict[name]
# Freeze the layers except the last 4 layers
for layer in pretrained_model.layers[:-4]:
    layer.trainable = False
         
# Check the trainable status of the individual layers
for layer in pretrained_model.layers:
    print(layer, layer.trainable)

#name_space = name_dict[name]
model.add(pretrained_model)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
 
 # Show a summary of the model. Check the number of trainable parameters
model.summary()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
 # Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10
  
train_generator = train_datagen.flow(
        resize_x_train,
        y_train,
        #target_size=(image_size, image_size),
        batch_size=train_batchsize)
   
validation_generator = validation_datagen.flow(
        x_test,
        y_test,
        #target_size=(image_size, image_size),
        batch_size=val_batchsize,
        shuffle=False)

model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc'])
# Train the model

batch_size = 128
epochs = 12

history = model.fit(
         batch_size=batch_size,
         epochs=epochs,
         steps_per_epoch=10,
         verbose=1,
         validation_data=(x_test, y_test))
#history = model.fit_generator(
#        train_generator,
#        steps_per_epoch=train_generator.samples/train_generator.batch_size ,
#        epochs=30,
#        validation_data=validation_generator,
#        validation_steps=validation_generator.samples/validation_generator.batch_size,
#        verbose=1)
 
# Save the model
model.save('small_last4.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
  
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
   
plt.figure()
    
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
     
plt.show()
#img_path = ''
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = name_space.preprocess_input(x)
#
#features = model.predict(x)
#
#print(features)
