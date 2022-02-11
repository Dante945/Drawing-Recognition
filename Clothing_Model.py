import tensorflow as tf # TensorFlow
from keras.datasets import fashion_mnist # Contains all images of clothing for training and testing
from keras import layers # Layers of the neural network for feature extraction, processing and condensing
from keras.callbacks import EarlyStopping # Used for early stoppage

"""
Preprocessing the Data
1. Load the data set from keras datasets
2. Add another dimension to the data set in order to work with the model at hand
3. Turn given categorical variables into binary class, 1 if the category is present and 0 if it is not 
4. Normalize the data by changing the 255 rbg scale into a range of 0-1
"""
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

"""
Function to stop the training if it detects that the value loss hasnt improved over the past few epochs according
to the min_delta parameter which acts as a margin
"""
early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=5,
    restore_best_weights=True,
)


"""
Network:
Takes in a 28x28 image of a piece of clothing and using a stack of 2 convolution blocks is able to extract and filter
out the features that the model will use to classify an image to a category 10 different pieces of clothing

Preprocess layers-
In order to account for randomness of the drawn piece of clothing, I opted to add in random translations 
and movements to have a wider array of images of clothing

Convolution Block-
    2x Conv2D - contains the weights, specifies number of feature maps that is outputted,
    filter image for a specific feature
    Max Pool - used to condense the image in order to enhance the extracted features from the previous layer
    in this case travels across images by scanning every other pixel (pool_size)

Also includes a drop out layer which help with over-fitting since it eliminates a fraction of the input units
which in turn makes it harder for the model to pick up on sporadic and niche patterns that might not otherwise
apply to the validation data or the real world

Ends with a Flatten layer which transforms this data into a an interpretable 1 - dimension which is followed by
The last two Dense layers which can now take the final set of flattened data and make its prediction to 1 of 10 classes
"""
model = tf.keras.models.Sequential([

    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(1),
    layers.Dropout(0.5),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(1),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
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
    callbacks=[early_stopping]
    )

# Saves the model for future use
model.save('.h5')
print("Done!")
