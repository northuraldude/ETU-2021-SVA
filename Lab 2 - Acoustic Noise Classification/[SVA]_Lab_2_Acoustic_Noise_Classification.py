# <center>АНАЛИЗ ЗВУКА И ГОЛОСА</center>

**Преподаватель**: Рыбин Сергей Витальевич

**Группа**: 6304

**Студент**: Белоусов Евгений Олегович

## <center>Классификация акустических шумов</center>

*Необоходимый результат: неизвестно*


import os
import IPython
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

import seaborn as sns
# %matplotlib inline

from tqdm.notebook import tqdm

from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, TensorBoard)
from tensorflow.keras.layers import (Input, Dense, Convolution2D, BatchNormalization,
                                     Flatten, MaxPool2D, Activation)
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# from google.colab import drive
# drive.mount('/content/drive')

# Ручная часть работы - директория с набором аудиофайлов, набор меток классов, учёт разновидностей имён файлов
predictions = "predictions"
directory = "./content/drive/MyDrive/Training"

labels = ["background",
          "bags",
          "door",
          "keyboard",
          "knocking_door",
          "ring",
          "speech",
          "tool"]

num_classes = len(labels)

filename_search = {"background": ["background_"],
                   "bags": ["bags_", "bg_", "t_bags_"],
                   "door": ["door_", "d_", "t_door_"],
                   "keyboard": ["keyboard_", "t_keyboard_", "k_"],
                   "knocking_door": ["knocking_door_", "tt_kd_", "t_knocking_door_"],
                   "ring": ["ring_", "t_ring_"],
                   "speech": ["speech_"],
                   "tool": ["tool_"]}

# Параметры конфигурации для будущей модели нейросети
class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=7, n_classes=10, use_mfcc=True,
                 n_mfcc=20, n_folds=10, n_features=100, learning_rate=0.0001, max_epochs=50):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_features = n_features
        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length / 512)), 1)
        else:
            self.dim = (self.audio_length, 1)

# Извлечение метки аудиошума из названия аудиофайла
def get_label_from_filename(filename):
    for key, value in filename_search.items():
        for val in value:
            if (filename.find(val) == 0):
                return key

# Подготовка датафрейма
def prepare_dataframe(directory):
    files = ([f.path for f in os.scandir(directory) if f.is_file()])
    # Создание датафрейма по предоставленной в условии задачи схеме
    df = pd.DataFrame(columns=["filename", "label"])

    # Проход по всем аудиофайлам в наборе
    for path in tqdm(files[:]):
        filename = os.path.splitext(os.path.basename(path).strip())[0]
        label = get_label_from_filename(filename)
        
        # Добавляем обработанный аудиофайл в датафрейм
        row = pd.Series([filename, label], index = df.columns)
        df = df.append(row, ignore_index=True)
    
    return df

# Извлечение признаков из набора аудиофайлов
def prepare_data(config, directory, df):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    files = ([f.path for f in os.scandir(directory) if f.is_file()])

    # Задаём длительность аудиофайла
    input_length = config.audio_length

    i = 0
    # Проход по всем аудиофайлам в наборе
    for path in tqdm(files[:]):
        filename = os.path.splitext(os.path.basename(path).strip())[0]

        data, sr = librosa.load(path, sr=config.sampling_rate)

        # Обрезка/приведение длительности аудиофайла к указанной в параметрах конфигурации
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        
        # Извлечение признаков MFCC с помощью библиотеки librosa
        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
        i = i + 1

    return X

# Модель свёрточной нейросети
def get_2d_conv_model(config):
    num_classes = config.n_classes
    
    inp = Input(shape=(config.dim[0], config.dim[1], 1))
    
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    out = Dense(num_classes, activation=softmax)(x)
    
    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)
    model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), metrics=['acc'])
    
    return model

# Матрица ошибок классификации
def plot_confusion_matrix(predictions, y):
    max_test = y
    max_predictions = np.argmax(predictions, axis=1)
    matrix = confusion_matrix(max_test, max_predictions)
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True,
                linewidths = 0.1, fmt="d", cmap = 'YlGnBu');
    plt.title("Матрица ошибок классификации", fontsize = 15)
    plt.ylabel("Настоящий класс")
    plt.xlabel("Предсказанный")
    plt.show()

# Подготовим датафрейм
df = prepare_dataframe(directory)

df.head()

# Сериализуем датафрейм в целях дальнейшей экономии времени
df.to_pickle("./content/drive/MyDrive/SVA_lab_2_dataframe.pkl")

# Десериализация ранее сохранённого датафрейма
df = pd.read_pickle("./content/drive/MyDrive/SVA_lab_2_dataframe.pkl")

# Подсчёт количества аудиозаписей каждого класса
df["label"].value_counts()

# Представим значения меток классов в виде целых чисел
encode = LabelEncoder()
encoded_labels = encode.fit_transform(df['label'].to_numpy())
df = df.assign(label=encoded_labels)

df.head()

# Задаём параметры конфигурации
config = Config(n_classes=num_classes, n_folds=10, n_mfcc=20)

X_train = prepare_data(config, directory, df)
print(X_train.shape)

# Нормализация данных

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std

X_train

# ПРОВЕРКА НА ТЕСТОВОМ НАБОРЕ ДАННЫХ

files = ([f.path for f in os.scandir("./content/drive/MyDrive/Test") if f.is_file()])
# Создание датафрейма по предоставленной в условии задачи схеме
submission = pd.DataFrame(columns=["fname"])

# Проход по всем аудиофайлам в наборе
for path in tqdm(files[:]):
    filename = os.path.splitext(os.path.basename(path).strip())[0]
    
    # Добавляем имя аудиофайла в датафрейм
    row = pd.Series([filename], index = submission.columns)
    submission = submission.append(row, ignore_index=True)

submission.head()

X_test = prepare_data(config, "./content/drive/MyDrive/Test", submission)

# Нормализация данных

mean = np.mean(X_test, axis=0)
std = np.std(X_test, axis=0)

X_test = (X_test - mean)/std

X_test

if not os.path.exists(predictions):
    os.mkdir(predictions)
if os.path.exists("./content/drive/MyDrive/" + predictions):
    shutil.rmtree("./content/drive/MyDrive/" + predictions)

# Для кросс-валидации используется StratifiedKFold - разновдность KFold алгоритма, которая возвращает
# стратифицированные папки c данными: каждый набор в папке содержит примерно такой же процент выборок каждого целевого класса,
# что и полный набор.
skf = StratifiedKFold(n_splits=config.n_folds)
y_train = df["label"].values
y_train = np.stack(y_train[:])
model = get_2d_conv_model(config)
i = 0
for train_split, val_split in skf.split(X_train, y_train):
    K.clear_session()
    
    # Разделение имеющегося набора данных на тренировочную и валидационные выборки
    X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
    
    # Callback-функции для модели Keras
    # В ходе обучения сохраняем веса лучшей модели для потенциального дальнейшего использования
    checkpoint = ModelCheckpoint('best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    print("#"*50)
    print("Fold: ", i)
    model = get_2d_conv_model(config)
    history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list, batch_size=256, epochs=config.max_epochs)
    model.load_weights('best_%d.h5'%i)
    
    # Сохраняем предсказания модели по тренировочным данным
    print("TRAIN PREDICTIONS: ", i)
    predictions = model.predict(X_train, batch_size=256)
    save_train_preds_path = "./predictions/train_predictions_{:d}.npy".format(i)
    np.save(save_train_preds_path, predictions)
    plot_confusion_matrix(predictions, y_train)
    
    # Сохраняем предсказания модели по тестовым данным
    print("TEST PREDICTIONS: ", i)
    predictions = model.predict(X_test, batch_size=256)
    save_test_preds_path = "./predictions/test_predictions_{:d}.npy".format(i)
    np.save(save_test_preds_path, predictions)
    
    j = 0
    for prob in predictions:
        #print(prob)
        #print(np.argmax(prob))
        submission.loc[j,'score'] = max(prob)
        prob_index = list(prob).index(max(prob))
        #print(prob_index)
        submission.loc[j,'label'] = prob_index
        j += 1

    submission_result = submission.copy()
    submission_result['label'] = encode.inverse_transform(np.array(submission['label']).astype(int))
    submission = submission_result
    save_submission_path = "./predictions/submission_{:d}.npy".format(i)
    submission.to_csv(save_submission_path.format(i), index=False)
    
    i += 1

