import tensorflow as tf  # TensorFlow
from keras.datasets import mnist  # Contains all images of digits for training and testing
from keras import layers  # Layers of the neural network for feature extraction, processing and condensing

"""
Preprocessing the Data
1. Load the data set from keras datasets
2. Add another dimension to the data set in order to work with the model at hand
3. Turn given categorical variables into binary class, 1 if the category is present and 0 if it is not 
4. Normalize the data by changing the 255 rbg scale into a range of 0-1
"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

"""
Network:
Takes in a 28x28 image of a hand drawn digit and using a convolution block is able to extract and filter
out the features that the model will use to classify an image to a category 0-9

Approach: 
I wanted to go with a much more simple approach compared to the clothing model since were dealing with much more simple
data, numbers vs. entire pieces of clothing

Preprocess layers-
In order to account for randomness of the drawn digit, I opted to add in random translations 
and movements to have a wider array of possible numbers drawn by a human

Convolution Block-
    Conv2D - contains the weights, specifies number of feature maps that is outputted,
    filter image for a specific feature
    Max Pool - used to condense the image in order to enhance the extracted features from the previous layer
    in this case travels across images by scanning every other pixel (pool_size)

Ends with a Flatten layer which transforms this data into a an interpretable 1 - dimension which is followed by
The last two Dense layers which can now take the final set of flattened data and make its prediction to 1 of 10 classes
"""
model = tf.keras.models.Sequential([

    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomRotation(factor=0.2),

    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dropout(0.5),

    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

"""
Compilation:
adam- general optimizer functions which will help the model learn by adjusting the weights of the
network to minimize the loss (difference between guess and known answer)
categorical_crossentropy - used to measure the loss with multiple categories involved and not just "yes" or "no"
"""
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""
Fitting:
passes in both training and validation set to begin the learning process of the relationship between the image and its class
batch size - number of iterations of training data
epochs - a complete round of training
"""
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=128,
    epochs=10,
)

# Saves the model for future use
model.save('Digit_Weights.h5')
print("Done!")
