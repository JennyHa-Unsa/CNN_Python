import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Paso 1: Cargar los datos (se asume que los archivos CSV tienen encabezado)
archivo_train_path = 'database/HAM10000/HAM10000_train.csv'
archivo_test_path = 'database/HAM10000/HAM10000_test.csv'
train_data = pd.read_csv(archivo_train_path)
test_data = pd.read_csv(archivo_test_path)


# Mostrar las primeras filas para verificar los encabezados
print(train_data.head())
print(test_data.head())

# Extraer las imágenes (características) y las etiquetas
X_train = train_data.drop(columns=['label']).values  # Eliminar la columna 'label'
y_train = train_data['label'].values  # La columna 'label' son las etiquetas
X_test = test_data.drop(columns=['label']).values  # Eliminar la columna 'label'
y_test = test_data['label'].values  # Las etiquetas del conjunto de prueba

# Normalizar las imágenes a valores entre 0 y 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape de las imágenes a formato 60x23x1 para que TensorFlow las entienda
X_train = X_train.reshape(-1, 60, 23, 1)
X_test = X_test.reshape(-1, 60, 23, 1)

# Convertir las etiquetas a formato categórico
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# División de los datos de entrenamiento en un conjunto de validación (si es necesario)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Paso 2: Construcción del modelo CNN
model = Sequential()

# Capa convolucional 1 (ajustada al nuevo tamaño de imagen)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(60, 23, 1)))
model.add(MaxPooling2D((2, 2)))

# Capa convolucional 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Aplanar las salidas de las capas convolucionales
model.add(Flatten())

# Capa densa (fully connected)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Regularización con Dropout

# Capa de salida
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Paso 3: Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Paso 4: Evaluación del modelo
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular precisión, recall y F1-score
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Obtener Accuracy directamente del modelo
accuracy = history.history['accuracy'][-1]  # Último valor de Accuracy durante el entrenamiento

# Imprimir todas las métricas
print(f'\nMETRICAS DEL MODELO:\n-------------------------')
print(f'* Accuracy: {accuracy:.4f}')
print(f'* Precisión: {precision:.4f}')
print(f'* Recall: {recall:.4f}')
print(f'* F1-Score: {f1:.4f}')

# Paso 5: Visualización
# Graficar precisión
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

# Graficar pérdida
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

#Paso 6_ Guardar el modelo 
model.save('ham10000_cnn_model.h5')
