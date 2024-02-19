"""
Bu modül hedef veri setine uygulanacak işlemlerin scriptini içerir.

*** UYARI ***
    Bu script sadece "Breast Cancer Wisconsin (Diagnostic) Data Set" için tasarlanmıştır.
"""

import additional_funcs
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix


class Script:
    def __init__(self):
        self.dataframe = None
        self.uploaded_file = None
        self.preprocessed_dataframe = None
        self.classifier_name = None
        self.classifier = None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_preds = None
        self.cf_matrix = None

    def data_preprocess(self, dataframe):
        """
        Veri setinin ön işleme işlemlerini gerçekleştirir.
        :param dataframe:
        :return:
        """
        new_dataframe = dataframe.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)  # 'id' ve 'Unnamed: 32' sütunları siliniyor.

        # Verinin son 10 satırı
        # new_dataframe[-10:]

        # Veri setinde 0 olan niteliklerin hatalı olduğu düşünülmüştür.
        # Bu veriler sütunlarının ortalamalarıyla dolduruluyor.
        new_dataframe = additional_funcs.fill_mean_of_column_to_zero_values(dataframe)

        # Diagnosis sütununda 'M' degeri 1, 'B' degeri 0 olarak degistiriliyor
        new_dataframe['diagnosis'] = new_dataframe['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

        self.preprocessed_dataframe = new_dataframe

        # X ve y verileri esitleniyor
        self.y = np.array(new_dataframe['diagnosis'])
        self.X = new_dataframe.drop('diagnosis', axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def get_classifier(self):
        """
        Sidebardan seçilen modeli döndürür.
        :return:
        """
        if self.classifier_name == 'SVM':
            return SVC()
        elif self.classifier_name == 'KNN':
            return KNeighborsClassifier()
        else:
            return GaussianNB()

    def learn_with_bayes(self):
        """
        Naive Bayes eğitimi yapar.
        :return:
        """
        model = self.get_classifier()

        return model

    def gridsearch(self):
        """
        KNN ve SVM algoritmalarında sisteme uygun olan parametreleri bulur ve en verimli modeli döndürür.
        :return:
        """
        model = self.get_classifier()

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # define params
        knn_params = {
            'n_neighbors': range(1, 30),
            'weights': ['uniform', 'distance'],
            'leaf_size': range(1, 50, 5)
        }
        svm_params = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }

        if self.classifier_name == 'KNN':
            gridsearch = GridSearchCV(estimator=model, param_grid=knn_params, cv=10)
        else:
            gridsearch = GridSearchCV(estimator=model, param_grid=svm_params, cv=10)

        grid_result = gridsearch.fit(X_train, y_train)
        best_model = model.set_params(**grid_result.best_params_)

        return best_model

    def evaluate_model(self, model):
        """
        Modelin eğitimini yapar.
        :param model:
        :return:
        """
        model.fit(self.X_train, self.y_train)

        self.y_preds = model.predict(self.X_test)

        self.cf_matrix = confusion_matrix(self.y_test, self.y_preds)
