# # <center>АНАЛИЗ ЗВУКА И ГОЛОСА</center>
# 
# **Преподаватель**: Рыбин Сергей Витальевич
# 
# **Группа**: 6304
# 
# **Студент**: Белоусов Евгений Олегович

# ## <center>Классификация произвольных звуков</center>
# ### <center>(Kaggle's Freesound General-Purpose Audio Tagging Challenge)</center>
# 
# *Необоходимый результат: больше 0.76 (10 баллов), не меньше 0.7 (8 баллов), не меньше 0.33 (6 баллов)*

import warnings
warnings.filterwarnings('ignore')

import os
import shutil

import IPython
import numpy as np
import pandas as pd
import librosa

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, TensorBoard)

from tensorflow.keras.layers import (Input, Dense, Convolution2D, BatchNormalization,
                                     Flatten, MaxPool2D, Activation)
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras import backend as K


np.random.seed(1001)


# Загружаем датасет
train = pd.read_csv("./SVA/train.csv")
test = pd.read_csv("./SVA/sample_submission.csv")

train.head()
test.head()

# Базовая информация о датасете
print("Количество тренировочных сэмплов: ", train.shape[0])
print("Количество классов тренировочных сэмплов: ", len(train.label.unique()))

# Параметры конфигурации для будущей модели нейросети
class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length / 512)), 1)
        else:
            self.dim = (self.audio_length, 1)


LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])


# Модель свёрточной нейросети
def get_2d_conv_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    
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
    
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


# Первичная обработка данных
def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        # Загрузка аудиофайла
        print(i)
        print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

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
    return X


# Задаём параметры: частота дискретизации, длительность звукового файла, количество папок для кросс-валидации,
#  использование MFCC, количество признаков MFCC
config = Config(sampling_rate=44100, audio_duration=2, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=40)

X_train = prepare_data(train, config, './SVA/audio_train/')
X_test = prepare_data(test, config, './SVA/audio_test/')

X_train
X_test

# Приводим метки классов в виде чисел
y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

# Нормализация данных
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

PREDICTION_FOLDER = "predictions_2d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)
if os.path.exists('./logs/' + PREDICTION_FOLDER):
    shutil.rmtree('./logs/' + PREDICTION_FOLDER)

# Для кросс-валидации используется StratifiedKFold - разновдность KFold алгоритма, которая возвращает
# стратифицированные папки c данными: каждый набор в папке содержит примерно такой же процент выборок каждого целевого класса,
# что и полный набор.
skf = StratifiedKFold(n_splits=config.n_folds)
i = 0
for train_split, val_split in skf.split(X_train, train.label_idx):
    K.clear_session()
    X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
    # В ходе обучения сохраняем веса лучшей модели для потенциального дальнейшего использования
    checkpoint = ModelCheckpoint('best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    tb = TensorBoard(log_dir='.logs\\' + PREDICTION_FOLDER + '\\fold_%i'%i, write_graph=True)
    callbacks_list = [checkpoint, early, tb]
    print("#"*50)
    print("Fold: ", i)
    model = get_2d_conv_model(config)
    history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list, 
                        batch_size=64, epochs=config.max_epochs)
    model.load_weights('best_%d.h5'%i)

    # Сохраняем предсказания модели по тренировочным данным
    predictions = model.predict(X_train, batch_size=64, verbose=1)
    np.save(PREDICTION_FOLDER + "\\train_predictions_%d.npy"%i, predictions)

    # Сохраняем предсказания модели по тестовым данным
    predictions = model.predict(X_test, batch_size=64, verbose=1)
    np.save(PREDICTION_FOLDER + "\\test_predictions_%d.npy"%i, predictions)

    # Создание файла с результатами (submission)
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "\\predictions_%d.csv"%i)
    i += 1

# Сбор результатов из всех записанных файлов
pred_list = []
for i in range(10):
    pred_list.append(np.load("./predictions_2d_conv/test_predictions_%d.npy"%i))
prediction = np.ones_like(pred_list[0])
for pred in pred_list:
    prediction = prediction*pred
prediction = prediction**(1./len(pred_list))
# Создание файла с результатами (submission) для платформы Kaggle
top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]
test = pd.read_csv('./SVA/sample_submission.csv')
test['label'] = predicted_labels
test[['fname', 'label']].to_csv("2d_conv_ensembled_submission.csv", index=False)
