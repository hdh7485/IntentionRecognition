from keras.applications import xception
from keras.applications import inception_resnet_v2
from keras.applications import resnet_v2
from keras.applications import mobilenet_v2
from keras.applications import densenet
from keras.preprocessing import image
import numpy as np

name = 'xception'
model_dict = { 'xception': xception.Xception(weights='imagenet', include_top=False),
        'inception_resnet_v2': inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False),
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
model = model_dict[name]
name_space = name_dict[name]


img_path = ''
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = name_space.preprocess_input(x)

features = model.predict(x)

print(features)
