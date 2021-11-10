import tensorflow as tf
from cnn_utils import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))

idx=0
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: {}, sample: {}".format(str(Y_train.shape), Y_train[idx]))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), input_shape=(64,64,3), padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal(), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((8,8)),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal(), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((4,4)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6)
    ])

model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')
#loss=tf.keras.losses.Categorical_Crossentropy())
model.fit(X_train, Y_train, epochs=15)
model.evaluate(X_test, Y_test)
model.summary()
