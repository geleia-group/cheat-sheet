
<h1>**Geleia Cheat Sheet**</h1>


<h2>**Keras**</h2>


Keras é uma biblioteca de aprendizado profundo poderosa e fácil de usar para Theano e TensorFlow, que fornece uma API de redes neurais de alto nível para desenvolver e avaliar modelos de aprendizado profundo.

<h3>**Import packages**</h3>

```
>>> import numpy as np

>>> import tensorflow as tf
```

<h3>**A Basic Example**</h3>

```
>>> from tf.keras.models import Sequential

>>> from tf.keras.layers import Dense

>>> data = np.random.random((1000, 100))

>>> labels = np.random.randint (2, size= (1000, 1))

>>> model = Sequential()

>>> model.add(Dense (32,


     activation='relu',


     input_dim=100)

>>> model.add(Dense (2, activation='sigmoid'))

>>> model.compile (optimizer='rmsprop',

 		         loss='binary_cross_entropy',


                 metrics=['accuracy'])

>>> model. fit (data, labels, epochs=10, batch_size=32)

>>> predictions = model.predict(data)
```

<h3>**Data**</h3>

Seus dados precisam ser armazenados como matrizes Numpy ou como uma lista de matrizes NumPy. 

Idealmente, você divide os dados em conjuntos de treinamento e teste, para os quais também pode recorrer ao módulo **train_test_split** do sklearn.**cross_validation**.


<h4>**Keras Data Sets**</h4>

```
>>> from tf.keras.datasets import boston_housing, mnist, cifar10, imdb

>>> (X_train, y_train), (x_testry_test) = mist.load_data()

>>> (X_train2, y_train2), (X_test2, y_test2) = boston_housing.load_data()

>>> (X_train3, y_train3), (X_test3, y_test3) = cifario.load_data)

>>> (X_train4, y_train4), (X_test4,y_test4) = imdb.load_data (num_words=20000)

>>> num_classes = 10

```

<h4>**Other**</h4>

```
>>> import numpy

>>> data = numpy.loadtxt('http://bit.ly/pimadiabetes', delimiter=",")

>>> X = data[:,0:8]

>>> y = data [:,8]
```

<h3>**Preprocessing**</h3>


<h4>**Sequence Padding**</h4>

```
>>> from tf.keras.preprocessing import sequence

>>> X_train5 = sequence.pad_sequences(X_train4, maxlen=80)

>>> X_test4 = sequence.pad_sequences(X_test4, maxlen=80)
```

<h4>**One-Hot Encoding**</h4>

```
>>> from tf.keras.utils import to_categorical

>>> Y_train = to_categorical(y_train, num_classes)

>>> Y_test = to_categorical(y_test, num_classes)

>>> Y_train3 = to_categorica(y_train3, num_classes)

>>> Y_test3 = to_categorical (y_test3, num_classes)
```

<h4>**Train and Test Sets**</h4>

```
>>> from sklearn.model_selection import train_test_split

>>> X_train5, X_test5,y_train5,y_test5 = train_test_split(test_size=0.33, random_state=42)
```

<h4>**Standardization/Normalization**</h4>

```
>>> from sklearn.preprocessing isport 3 StandardScaler

>>> scaler = StandardScaler().fit(X_train2)

>>> standardized_X = scaler.transform(X_train2)

>>> standardized_X_test = scaler.transform/X_test2)
```

<h4>**Image Generator**</h4>

```
>>> from tensorflow.keras.preprocessing.image import ImageDataGenerator

>>> image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees

                               zoom_range=0.05, # Zoom in by 5% max

                               fill_mode='nearest') # Fill in missing pixels with the nearest filled value
```

<h3>**Model Architecture**</h3>

```
>>> from tf.keras.models import Sequential 

>>> from tf.keras.layers import Dense 

>>> from tf.keras.layers import Dropout 

>>> from tf.keras.layers import Flatten 

>>> from tf.keras.layers import Conv2D

>>> from tf.keras.layers import MaxPooling2D 

>>> from tf/keras.layers import Embedding, LSTM
```

<h4>**Sequential Model**</h4>

```
>>> from keras models import Sequential

>>> model = Sequential()

>>> model2 = Sequential)

>>> model3 = Sequential()
```

<h4>**Multilayer Perceptron (MLP)**</h4>

<h5>**Binary Classification**</h5>

```
>>> from keras.layers import Dense

>>> model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))

>>> model.add(Dense (8, kernel_initializer='uniform', activation='relu'))

>>> model.add(Dense (1, kernel_initializers'uniform', activation=' sigmoid)) 
```

<h5>**Multi-Class Classification**</h5>

```
>>> model.add(Dense(512, activation='relu', input_shape=(784,)))

>>> model.add(Dropout(0.2))

>>> model.add(Dense(512, activation='relu'))

>>> model.add(Dropout(0.2))

>>> model.add(Dense(10, activation='softmax'))
```

<h5>**Regression**</h5>

```
>>> model.add(Dense (64, activation='relu', input_dim-train_data.shape [1]))

>>> model.add(Dense(1)
```

<h5>**Convolutional Neural Network (CNN)**</h5>

```
>>> model2.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train shape [1:]))

>>> model2.add(activation ('relu'))

>>> model2.add(Conv2D(32, (3, 3)))

>>> model2.add(activation ('relu'))

>>> model2.add(MaxPooling2D(pool_size=(2, 2)))

>>> model2.add(Dropout(0.25))

>>> model2.add(Conv2D(64, (3, 3), padding='same'))

>>> model2.add(activation ('relu'))

>>> model2.add(Conv2D(64, (3, 3)))

>>> model2.add(activation ('relu'))

>>> model2.add(MaxPooling2D(pool_size=(2,2)))

>>> model2.add(Dropout(0.25)

>>> model2.add(Flatten())

>>> model2.add(Dense(512))

>>> model2.add(activation ('relu'))

>>> model2.add(Dropout(0.5))

>>> model2.add(Dense (num_classes)

>>> model2.add(activation''softmax'))
```

<h5>**Recurrent Neural Network (RNN)**</h5>

```
>>> model3.add(Embedding (20000, 128)

>>> model3.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

>>> model3.add(Dense(1, activation='sigmoid'))
```

<h3>**Keras Applications**</h3>

<h4>**Transfer learning**</h4>

```
# Instancie um modelo de visão treinado
>>> modelo_base = keras.applications.ResNet50()
```

<h4>**Freeze Layers**</h4>

```
# Freeze all layers

for layer in modelo_base.layers:

  layer.trainable = False

# Check the trainable status of the individual layers

for layer in modelo_base.layers:

  print(layer, layer.trainable)
```

<h4>**Add model**</h4>

```
# Cria a sequência de camadas

add_model = Sequential()

# Adiciona primeiro o modelo base

add_model.add(modelo_base)

# Precisamos de uma camada global de Pooling

add_model.add(GlobalAveragePooling2D())

# Dropout para regularização e evitar overfitting

add_model.add(Dropout(0.5))

# Camada densa na camada final com ativação softmax para previsão das probabilidades das classes

add_model.add(Dense(num_classes, activation = 'softmax'))
```

<h3>**Inspect Model**</h3>

```
Pode ser feito no model ou add_model:

>>> model.output_shape #Model output shape

>>> model. summary() #Model summary representation

>>> model.get_config() #Model configuration

>>> model.get_weights() #List all weights tensors in the model
```

<h3>**Compile Model**</h3>

<h4>**MLP: Binary Classification**</h4>

```
>>> model.compile(optimizer='adam',

loss='binary_crossentropy',

metrics=['accuracy'])
```

<h4>**MLP: Multi-Class Classification**</h4>

```
>>> model.compile(optimizer='rmsprop',

loss='categorical_crossentropy',

metrics=['accuracy'])
```

<h4>**MLP: Regression**</h4>

```
>>> model.compile (optimizer='rmsprop',

loss='mse',

metrics=['mae'])
```

<h4>**Recurrent Neural Network**</h4>

```
>>> model3.compile(loss='binary_crossentropy',

optimizer='adam',

metrics=['accuracy'])

<h3>**Model Training**</h3>


>>> model3.fit (X_train,Y_train,

batch_size=32,

epochs=15,

verbose=1,

validation_data=(x_test4, y_test4))
```

<h4>**From Image Generator**</h4>

```
>>> image_gen.fit(X_train)

>>> results = add_model.fit_generator(image_gen.flow(X_train, Y_train), epochs=100,


            validation_data=(X_teste, Y_teste), callbacks = [reduce_lr])
```

<h3>**Evaluate Your Model's Performance**</h3>

```
>>> score = model3.evaluate(X_test, batch_size=32)
```

<h3>**Prediction**</h3>

```
>>> model3.predict(X_test4, batch_size=32)

>>> model3.predict_classes (X_test4, batch_size=32)
```

<h3>**Save/ Reload Models**</h3>

```
>>> from tf.keras.models import load_model

>>> model3.save('model_file.h5')

>>> my_model = load_model('my_model.h5')
```

<h3>**Model Fine-tuning**</h3>
<h4>**Optimization Parameters**</h4>

```
>>> from tf.keras.optimizers import RM3prop

>>> opt = RM3prop (lr=0.0001, decay=le-6)

>>> model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

<h4>**Early Stopping**</h4>

```
>>> from tf.keras.callbacks import Earlystopping

>>> early_stopping_monitor = EarlyStopping (patience=2)

>>> model3.fit (X_train4,y_train4,batch_size=32,epochs=15, validation_data= {X_test4, y_test4), callbacks=(early_stopping_monitor))
```

<h4>**Reduce Learning Rate**</h4>

```
# Regra para a redução da taxa de aprendizado

reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
```

<h2>**Referências**</h2>
*   Datacamp
*   Keras.io