from pylab import *
from pandas import read_csv
import requests, zipfile, StringIO
import random

# Read the data into a pandas DataFrame.

#loan_stats_data = read_csv("tps://resources.lendingclub.com/LoanStats3b.csv.zip")
#loan_stats_data.head()


class LoanData:
    """
    """
    def __init__(self, csvfilename=None):
        if csvfilename:
            self.loan_data = read_csv(csvfilename, skiprows=1, skipfooter=2)
        else:
            self.requested_file = requests.get("https://resources.lendingclub.com/LoanStats3b.csv.zip")
            self.zipped = zipfile.ZipFile(StringIO.StringIO(self.requested_file.content))
            self.loan_data = read_csv(self.zipped.open('LoanStats3b.csv'), skiprows=1, skipfooter=2)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.loan_data_numeric = self.loan_data.select_dtypes(include=numerics)
        self.loan_data_nonnumeric = self.loan_data.select_dtypes(exclude=numerics)

    def summary_statistics(self, column_name):
        if not column_name in self.loan_data:
            raise Exception('invalid column name')
        description = self.loan_data[column_name].describe()
        if description['unique'] <= 100:
            description['unique_values'] = unique(self.loan_data[column_name].get_values())
            description['value_counts'] = self.loan_data[column_name].value_counts().to_dict()
        if column_name in self.loan_data_numeric:
            description['count_missing'] = self.loan_data['mths_since_last_major_derog'].isnull().sum()
        return description

    def get_columns(self):
        return self.loan_data.columns

    def head(self):
        return self.loan_data.head()

    def save_histogram(self, column_name, filename):
        if column_name in self.loan_data_nonnumeric:
            raise Exception('Cannot generate histogram for non-numeric data')
        figure(figsize=(12, 9))
        ax = subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        xticks(fontsize=14)
        yticks(range(5000, 30001, 5000), fontsize=14)
        xlabel(column_name, fontsize=16)
        ylabel("Count", fontsize=16)
        hist(list(self.loan_data[column_name].values) + list(self.loan_data[column_name].values),
                color="#3F5D7D", bins=100)
        text(1300, -5000, "Data source: Lending Circle | "
               "Author: Ramesh Thulasiram", fontsize=10)
        savefig(filename or (column_name+'.png'), bbox_inches="tight")


    def save_bar_chart(self, column_name, filename):
        if column_name in self.loan_data_numeric:
            raise Exception('Cannot generate bar chart for numeric data')
        #figure(figsize=(12, 9))
        figure()
        ax = subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        #xticks(fontsize=14)
        yticks(range(5000, 30001, 5000), fontsize=14)
        xlabel(column_name, fontsize=16)
        ylabel("Count", fontsize=16)
        width = 0.35
        bar_dict = self.loan_data[column_name].value_counts().to_dict()
        bar(arange(len(bar_dict.keys())), bar_dict.values(), width,  color="#3F5D7D")
        xticks(arange(len(bar_dict.keys()))+width/2., bar_dict.keys() )
        text(1300, -5000, "Data source: Lending Circle | "
               "Author: Ramesh Thulasiram", fontsize=10)
        savefig(filename or (column_name+'.png'))

    def get_random_1000(self):
        ran1000 = random.sample(self.loan_data.index, 1000)
        return self.loan_data[ran1000]

    def runall(self):
        pass #from q1 
