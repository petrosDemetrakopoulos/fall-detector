from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def plot_sensor_signal(data, snesor_name='Accelerometer'):
    plt.plot(data['seconds_elapsed'],data['x'], label='X')
    plt.plot(data['seconds_elapsed'],data['y'], label='Y')
    plt.plot(data['seconds_elapsed'],data['z'], label='Z')
    plt.title(snesor_name + ' Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Sensor reading')
    plt.legend()
    plt.show()

def read_data():
    accelerometer_data = pd.read_csv('./data/Accelerometer.csv',header=0, names=['time','seconds_elapsed','z_acc','y_acc','x_acc'])
    gyro_data = pd.read_csv('./data/Gyroscope.csv',header=0, names=['time','seconds_elapsed','z_gyro','y_gyro','x_gyro'])
    gravity_data = pd.read_csv('./data/Gravity.csv',header=0, names=['time','seconds_elapsed','z_gravity','y_gravity','x_gravity'])

    # drop columns that are common in all 3 dataframes and keep only the ones from the accelerometer dataframe
    gyro_data = gyro_data.drop(['time','seconds_elapsed'],axis=1)
    gravity_data = gravity_data.drop(['time','seconds_elapsed'],axis=1)

    # concatenate readings from all sensors to a single dataframe
    data = pd.concat([accelerometer_data,gyro_data,gravity_data],axis=1)
    return data

def normalize(data):
    data_norm = data.copy()
    columns_for_normalization = ['x_acc','y_acc','z_acc','x_gravity', 'y_gravity', 'z_gravity', 'x_gyro', 'y_gyro', 'z_gyro']
    for column in columns_for_normalization:
        data_norm[column] = (data_norm[column] - data_norm[column].mean()) / data_norm[column].std()
    
    return data_norm

def feature_extraction(signal_df, window_size=100, overlap=50):
    num_components = signal_df.shape[1]
    num_samples = signal_df.shape[0]

    # used to rename columns in the extracted features (mentioning the statistical moment)
    new_col_names_mean = dict()
    new_col_names_std = dict()
    new_col_names_skewness = dict()
    new_col_names_kurtosis = dict()

    for col in signal_df.columns:
        new_col_names_mean[col] = col + '_mean'
        new_col_names_std[col] = col + '_std'
        new_col_names_skewness[col] = col + '_skewness'
        new_col_names_kurtosis[col] = col + '_kurtosis'
    
    # Initialize empty DataFrames to store results
    mean_df = pd.DataFrame(columns=new_col_names_mean.values())
    std_df = pd.DataFrame(columns=new_col_names_std.values())
    skewness_df = pd.DataFrame(columns=new_col_names_skewness.values())
    kurtosis_df = pd.DataFrame(columns=new_col_names_kurtosis.values())

    # calculate number of total windows
    num_total_windows = ((num_samples - window_size) // (window_size - overlap)) + 1

    # Calculate statistical moments for each window
    for i in range(num_total_windows):
        window_start = i*(window_size - overlap)
        window = signal_df.iloc[window_start:window_start + window_size]

        # Calculate statistical moments for each component
        window_mean = window.mean().rename(new_col_names_mean)
        window_std = window.std().rename(new_col_names_std)
        window_skewness = window.apply(skew).rename(new_col_names_skewness)
        window_kurtosis = window.apply(kurtosis).rename(new_col_names_kurtosis)

        # Append the results to the respective DataFrames
        mean_df = pd.concat([mean_df, pd.DataFrame([window_mean], columns=new_col_names_mean.values())], ignore_index=True)
        std_df = pd.concat([std_df, pd.DataFrame([window_std], columns=new_col_names_std.values())], ignore_index=True)
        skewness_df = pd.concat([skewness_df, pd.DataFrame([window_skewness], columns=new_col_names_skewness.values())], ignore_index=True)
        kurtosis_df = pd.concat([kurtosis_df, pd.DataFrame([window_kurtosis], columns=new_col_names_kurtosis.values())], ignore_index=True)

    # Combine the results into a single DataFrame
    extracted_features = pd.concat([mean_df, std_df, skewness_df, kurtosis_df], axis=1)
    
    return extracted_features

def plot_out_signal(predictions):
    plt.clf()
    plt.plot(predictions)
    plt.title('Predicted labels over time')
    plt.xlabel('Time (id of window)')
    plt.ylabel('Predicted label')
    plt.show()

def plot_pca(extracted_features, color=None):
    plt.clf()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(extracted_features)

    plt.scatter(reduced[:,0],reduced[:,1], c=color, cmap='bwr')
    title = 'Extracted Features (PCA, n_components=2)' + (' - Predicted labels' if color is not None else '')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def main():
    data = read_data()
    normalized_data = normalize(data)

    # time related columns are not needed after this point
    normalized_data = normalized_data.drop(['time','seconds_elapsed'],axis=1)
    extracted_features = feature_extraction(normalized_data)

    # dimensionality reduction using PCA
    plot_pca(extracted_features)

    # fit K-Means
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(extracted_features)
    predictions = kmeans.predict(extracted_features)

    # plot the results
    plot_pca(extracted_features, predictions)
    plot_out_signal(predictions)

if __name__ == "__main__":
    main()
