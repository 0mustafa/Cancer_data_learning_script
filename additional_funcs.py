"""
Bu modül hedef veri setinin scriptine yardımcı olacak fonksiyonları içerir.

*** UYARI ***
    Bu fonksiyonlar sadece "Breast Cancer Wisconsin (Diagnostic) Data Set" için tasarlanmıştır.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def fill_mean_of_column_to_zero_values(dataframe):
    """
    Veri setinde 0 nitelikler sütunlarının ortalama değerleriyle doldurulur.
    :param dataframe:
    :return:
    """
    count_of_columns_zero_values = dict((dataframe == 0).sum())
    zero_values_dict = {key: value for key, value in filter(lambda item: item[1] != 0,
                                                            count_of_columns_zero_values.items())}

    for column_name in zero_values_dict.keys():
        mean_of_column = np.mean(dataframe[dataframe[column_name] != 0][column_name])
        dataframe[column_name] = dataframe[column_name].apply(lambda x: mean_of_column if x == 0 else x)

    return dataframe


def plot_correlation(dataframe):
    """
    Korelasyon matrisini döndürür.
    :param dataframe:
    :return:
    """
    malignant = dataframe[dataframe['diagnosis'] == 1]
    benign = dataframe[dataframe['diagnosis'] == 0]

    # korelasyon matrisi
    correlation_matrix = dataframe.corr()
    plot = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')

    return plot


def plot_scatter(dataframe):
    """
    Scatter grafiğini döndürür
    :param dataframe:
    :return:
    """
    malignant = dataframe[dataframe['diagnosis'] == 1]
    benign = dataframe[dataframe['diagnosis'] == 0]

    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=malignant, x='radius_mean', y='texture_mean', color='red', alpha=0.5, label='Kötü')
    sns.scatterplot(data=benign, x='radius_mean', y='texture_mean', color='green', alpha=0.5, label='İyi')
    plt.title("Radius Mean vs Texture Mean")
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.legend()

    return fig


def format_time(seconds):
    """
    Saniyeyi saat, dakika, saniye cinslerine çevirir.
    :param seconds:
    :return:
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return hours, minutes, seconds
