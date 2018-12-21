import os.path,sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../../')
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks 
N_CATEGORIES  = len(os.listdir('./image-data/train/'))
IMAGE_SIZE = 75
BATCH_SIZE = 16
EPOCH = 50

NUM_TRAINING = 400
NUM_VALIDATION = 100

input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)) # px px rgb
base_model = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(N_CATEGORIES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[:15]:
   layer.trainable = False
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   rotation_range=10)

validation_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
)

train_generator = train_datagen.flow_from_directory(
   'image-data/train/',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
   'image-data/validation/',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

#model = load_model('./model/flower.hdf5')
model.load_weights('./model/flower.hdf5')

hist = model.fit_generator(train_generator,
   steps_per_epoch=NUM_TRAINING//BATCH_SIZE,
   epochs=EPOCH,
   verbose=1,
   validation_data=validation_generator,
   validation_steps=NUM_VALIDATION//BATCH_SIZE,
   )

model.save('./model/flower.hdf5')
