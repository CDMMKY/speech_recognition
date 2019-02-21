import struct
from csv import reader
from os import path

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import cfg


def list_files(start, stop, format_file):
    print('\nGenerate list files:')
    files = []
    for speaker in range(1, cfg.COUNT_SPEAKERS + 1):
        for record in range(start, stop + 1):
            files.append(f"{speaker} ({record}).{format_file}")
        print('.', end='')
    return files


def read_file(dir_name=cfg.FEATURED_DIRECTORY, start=1, stop=cfg.COUNT_RECORDS_TRAINING):
    # field_size_limit(maxsize)  # иначе слишком большой csv
    format_file = 'htk' if cfg.IS_HTK else 'csv'
    files = list_files(start, stop, format_file)
    features = read_htk(files, dir_name) if cfg.IS_HTK else read_csv(files, dir_name)
    if cfg.IS_DIMENSIONALITY:
        features = dimensionality_reduction(features, cfg.DIMENSIONALITY)
    return features


def read_csv(files, dir_name):
    features = []
    for file in files:
        with open(path.join(dir_name, file), newline='') as csvfile:
            reader_csv = reader(csvfile, delimiter=';')
            next(reader_csv, None)  # можно попробовать сделать словарь
            for row in reader_csv:
                features.append([float(n) for n in row[2:]])
    return features


def read_htk(files, dir_name):
    features = []
    for file in files:
        fin = open(path.join(dir_name, file), "rb")
        data_bin = fin.read()
        nframes, frate, nbytes, feakind = struct.unpack('>iihh', data_bin[:12])
        ndim = int(nbytes / 4)  # feature dimension (4 bytes per value)
        features.append(struct.unpack('>' + 'f' * ndim, data_bin[12:]))
    return features


def generate_labels(records=cfg.COUNT_RECORDS_TRAINING):
    print('\nGenerate labels: ')
    labels = []
    speakers = cfg.COUNT_SPEAKERS
    for speaker in range(speakers):
        ones = [0] * (speakers * records - 1)
        ones[speaker * records:speaker * records + records - 1] = [1] * records
        labels.append(ones)
        print('.', end='')
    return labels


def dimensionality_reduction(X, dimensionality):
    pca = PCA(n_components=dimensionality)
    reduced = pca.fit_transform(X)
    # toto = pca.singular_values_
    return reduced


def training(X, target, classifier):
    print('\nTrain:')
    models = []
    clf = cfg.CLASSIFIER_MAP.get(classifier)
    if not clf:
        return
    for y in target:
        model = clf.fit(X, y)
        models.append(model)
        print('.', end='')
    return models


def test(models, X, y):  # data, target
    print('\nTest: ')
    scores_proba, scores_acc, scores_sc = [], [], []
    for model in models:
        predicted = model.predict(X)
        scores_proba.append(
            model.predict_proba(X))  # считает апостериорную вероятность возможных результатов для выборок в X.
        for _y in y:
            scores_acc.append(accuracy_score(y_true=_y, y_pred=predicted))  # expected - цель _y
            scores_sc.append(model.score(X, _y))
        print('.', end='')
    return [scores_proba, scores_acc, scores_sc]


if __name__ == '__main__':
    features_train = read_file(start=1, stop=cfg.COUNT_RECORDS_TRAINING)  # 40*20
    labels_learning = generate_labels(records=cfg.COUNT_RECORDS_TRAINING)  # 20*800
    models = training(features_train, labels_learning, 'svm-linear')
    features_test = read_file(start=cfg.COUNT_RECORDS_TRAINING + 1,
                              stop=cfg.COUNT_RECORDS_TRAINING + cfg.COUNT_RECORDS_TEST)  # 10*20
    labels_test = generate_labels(records=cfg.COUNT_RECORDS_TEST)  # 20*200
    scores = test(models, features_test, labels_test)
    print(scores)
