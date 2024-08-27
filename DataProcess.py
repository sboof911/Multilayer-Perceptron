import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class DataProcess:
    _DataFrame : pd.DataFrame
    _DataFrame_mean : pd.Series
    _DataFrame_std : pd.Series
    _df_train : pd.DataFrame
    _df_test : pd.DataFrame
    _sorted_corr_list : list[tuple]
    _overall_homogeneity : pd.Series
    _features_to_remove : set

    def __init__(self, path : str, pngPlot = False, classes = None) -> None:
        self.CostumizedreadFile(path)
        if classes is not None:
            classes = [classes] if isinstance(classes, str) else classes
            if not isinstance(classes, list):
                raise Exception(f"classes variable must be str or a list of strs and {type(classes)} is given!")
            if any(classs not in self._DataFrame.columns for classs in classes):
                raise Exception(f"One or more class columns: ==> {classes} <== not in the DataFrame columns. DataFrame columns: {self._DataFrame.columns}")
        self.CleanData(classes)
        self.StandardizeData(classes)
        self.plot_Data(pngPlot)
        self.SortedCorrelationFeatures(classes)
        self.get_top_homogeneous_features()

    def CleanData(self, classes):
        print("Cleaning Data")
        df = self._DataFrame.copy()
        if classes is not None:
            df = df.drop(columns=classes)
        df.fillna(df.mean(), inplace=True)
        df.drop_duplicates(inplace=True)
        if classes is not None:
            df[classes] = self._DataFrame[classes]
        self._DataFrame = df

    def StandardizeData(self, classes):
        print("Standarizing Data")
        df = self._DataFrame.copy()
        if classes is not None:
            df = df.drop(columns=classes)
        self._DataFrame_std = pd.Series(df.std(), name='std').to_frame().T
        self._DataFrame_mean = pd.Series(df.mean(), name='mean').to_frame().T
        df = (df - df.mean()) / df.std()
        if classes is not None:
            df[classes] = self._DataFrame[classes]
        self._DataFrame = df

    def plot_Data(self, pngPlot):
        if pngPlot:
            print("Saving pairplot Data")
            pairplot = sns.pairplot(self._DataFrame, hue='class', diag_kind='hist')
            print("Saving pairplot in a png file")
            pairplot.savefig("pairplot.png", dpi=300, bbox_inches='tight')
            print("png file saved!")
            plt.close(pairplot.figure)

    def SortedCorrelationFeatures(self, classes):
        print("Get Serted Correclation Features")
        df = self._DataFrame.copy()
        if classes is not None:
            df = df.drop(columns=classes)

        corr_matrix = df.corr()
        corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
        corr_pairs = corr_matrix.stack()
        unique_pairs = {(min(f1, f2), max(f1, f2)): corr for (f1, f2), corr in corr_pairs.items()}
        self._sorted_corr_list = [(f1, f2, corr) for (f1, f2), corr in sorted(unique_pairs.items(), key=lambda x: x[1], reverse=True)]

    def get_top_homogeneous_features(self):
        print("Get The Top homogenous features")
        mean_std = self._DataFrame.groupby('class').agg(['mean', 'std'])
        cv = mean_std.xs('std', level=1, axis=1) / mean_std.xs('mean', level=1, axis=1)
        homogeneity_scores = 1 - cv
        self._overall_homogeneity = homogeneity_scores.mean(axis=0)

    def get_top_homogeneous_features_n(self, top_hom_n: int = 10):
        print('-' * 80)
        print(f"==> Printing the top {top_hom_n} low homogenous features:")
        return self._overall_homogeneity.nsmallest(top_hom_n)

    def print_Results(self, numberoflines : int = 10):
        print(self.get_top_homogeneous_features_n(numberoflines))
        print('-' * 80)
        print(f"==> Printing the top {numberoflines} Correlated features:")
        for feature1, feature2, corr in self._sorted_corr_list[:numberoflines]:
            print(f"[{feature1} - {feature2}] : {corr}")
        print('-' * 80)

    def get_features_to_remove(self, corr_threshold=0.9, homogeneity_threshold=0.5):
        print("Identify features to remove based on correlation and homogeneity")
        removal_scores = {}
        for f1, f2, corr in self._sorted_corr_list:
            if corr >= corr_threshold:
                removal_scores[f1] = removal_scores.get(f1, 0) + 1
                removal_scores[f2] = removal_scores.get(f2, 0) + 1

        for feature, homogeneity in self._overall_homogeneity.items():
            if homogeneity < homogeneity_threshold:
                removal_scores[feature] = removal_scores.get(feature, 0) + 1

        return sorted(removal_scores.items(), key=lambda x: x[1], reverse=True)
    
    def SplitShuffle_df(self, train_size=80):
        shuffled_indices = np.random.permutation(self._DataFrame.index)
        df_shuffled = self._DataFrame.reindex(shuffled_indices).reset_index(drop=True)
        split_index = int((train_size / 100) * len(df_shuffled))

        self._df_train = df_shuffled.iloc[:split_index]
        self._df_test = df_shuffled.iloc[split_index:]

    def CostumizedreadFile(self, path):
        print(f"Reading csv data from {path}")
        self._DataFrame = pd.read_csv(path, header=None)
        print("Data loaded!")
        columns_name = [f'feature{i}' for i in range(1, self._DataFrame.shape[1])]
        columns_name.insert(1, "class")
        self._DataFrame.columns = columns_name
        print("Columns names defined!")

    def CostumRemoveFeatures(self, scoreToRemove : int = 4):
        features_to_remove = [feature for feature, score in self.get_features_to_remove(0.95, 0.5) if score > scoreToRemove]
        print(f"the features to remove are : {features_to_remove}")
        self._DataFrame.drop(columns=features_to_remove, inplace=True)

    def DataInFile(self, directory_path = "./PreProcessedData"):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        self._df_train.to_csv(directory_path + "/train.csv")
        print(directory_path + "/train.csv Created!")
        self._df_test.to_csv(directory_path + "/test.csv")
        print(directory_path + "/test.csv Created!")
        mean_std = pd.concat([self._DataFrame_mean, self._DataFrame_std])
        print(mean_std)
        mean_std.to_csv(directory_path + "/mean_std.csv")
        print(directory_path + "/mean_std.csv Created!")

if __name__ == "__main__":
    try:
        datafile = "./data.csv"
        if not os.path.exists(datafile):
            raise Exception("data.csv doesn t exist!")
        Dataprocess = DataProcess(path=datafile, classes="class")
        Dataprocess.print_Results(10)
        Dataprocess.CostumRemoveFeatures(4)
        Dataprocess.SplitShuffle_df(80)
        Dataprocess.DataInFile()
    except Exception as e:
        print(f"Error: {e}")
