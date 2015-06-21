from pylab import *
from pandas import read_csv
import requests, zipfile, StringIO
import random
import numpy as np

#loan_stats_data = read_csv("tps://resources.lendingclub.com/LoanStats3b.csv.zip")
#loan_stats_data.head()


class LoanData:
    """
    Prints summary statistics of columns, train and predict the model, get random 1000 rows
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
        if 'unique' in description and description['unique'] <= 100:
            description['unique_values'] = unique(self.loan_data[column_name].get_values())
            description['value_counts'] = self.loan_data[column_name].value_counts().to_dict()
        if column_name in self.loan_data_numeric:
            description['count_missing'] = self.loan_data['mths_since_last_major_derog'].isnull().sum()
        return description

    def get_columns(self):
        return self.loan_data.columns

    def head(self):
        return self.loan_data.head()

    def preprocess_data(self):
        import re

        self.loan_data['int_rate'] = self.loan_data['int_rate'].map(lambda x: x.rstrip('%'))
        self.loan_data['int_rate'] = self.loan_data['int_rate'].astype('float32')
        #self.loan_data['revol_util'] = self.loan_data['revol_util'].map(lambda x: x.rstrip('%'))
        #self.loan_data['revol_util'] = self.loan_data['revol_util'].astype('float32')
        self.loan_data['term'] = self.loan_data['term'].map(lambda x: x.rstrip(' months'))
        self.loan_data['term'] = self.loan_data['term'].astype('int16')
        self.loan_data['emp_length'] = self.loan_data['emp_length'].map(lambda x: re.sub('\D', '', x, ) or 0)
        self.loan_data['emp_length'] = self.loan_data['emp_length'].astype('int16')

    def convert_categoricals(self, categoricals):
        for cat in categoricals:
            self.loan_data[cat] = self.loan_data[cat].astype('category')

    def convert_date_fields(self, date_fields):
        from datetime import date
        mon = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        for datef in date_fields:
            self.loan_data[datef] = self.loan_data[datef].map(lambda x: date(int(x.split('-')[1]), mon[x.split('-')[0]] ,1).strftime('%s') )
            self.loan_data[datef] = self.loan_data[datef].astype('int32')

    def train_and_predict(self):
        import pandas as pd
        from sklearn import tree
        from sklearn.feature_extraction import DictVectorizer
        from sklearn import preprocessing
        from sklearn.ensemble import RandomForestClassifier

        drop_columns = [ 'id', 'member_id', 'emp_title', 'url', 'desc', 'title' , 'last_pymnt_d','next_pymnt_d','last_credit_pull_d',
                         'revol_util']
        categoricals = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'pymnt_plan', 'purpose',
                        'zip_code', 'addr_state', 'initial_list_status']
        date_fields = ['issue_d', 'earliest_cr_line']
        self.convert_categoricals(categoricals)
        self.convert_date_fields(date_fields)
        self.preprocess_data()

        for col in (set(self.loan_data.columns) - set(categoricals) - set(drop_columns)):
            self.loan_data[col].fillna(0, inplace=True)
        self.loan_data['is_train'] = np.random.uniform(0, 1, len(self.loan_data)) <= .75
        self.train, self.test = self.loan_data[self.loan_data['is_train']==True], self.loan_data[self.loan_data['is_train']==False]

        # Convert loan_status to a format that we can use
        labels_train = (self.loan_data[self.loan_data['is_train']==True])[['loan_status']]
        labels_test = (self.loan_data[self.loan_data['is_train']==False])[['loan_status']]
        le = preprocessing.LabelEncoder()
        dv = DictVectorizer(sparse=False)
        labels_train = le.fit_transform(labels_train)
        labels_test = le.transform(labels_test)

        # vectorize training data
        categorical_view = self.loan_data.drop(list( (set(self.loan_data.columns) - set(categoricals)) ) , axis=1)
        del self.loan_data['loan_status']

        categorical_train_as_dicts = [dict(r.iteritems()) for _, r in categorical_view[self.loan_data['is_train']==True].iterrows()]
        categorical_train_fea = dv.fit_transform(categorical_train_as_dicts)
        categorical_test_as_dicts = [dict(r.iteritems()) for _, r in categorical_view[self.loan_data['is_train']==False].iterrows()]
        categorical_test_fea = dv.transform(categorical_test_as_dicts)

        numerical_train = self.loan_data[self.loan_data['is_train']==True].drop(list( set(drop_columns) | set(categoricals) | set(['is_train'])), axis=1)
        numerical_train_fea = numerical_train.as_matrix()
        numerical_test = self.loan_data[self.loan_data['is_train']==False].drop(list( set(drop_columns) | set(categoricals) | set(['is_train'])), axis=1)
        numerical_test_fea = numerical_test.as_matrix()

        train_fea = np.concatenate( (categorical_train_fea, numerical_train_fea), axis=1)
        test_fea = np.concatenate( (categorical_test_fea, numerical_test_fea), axis=1)

        self.clf = RandomForestClassifier(n_jobs=10)
        self.clf.fit(train_fea, labels_train)
        label_predictions = le.inverse_transform(self.clf.predict(test_fea).astype('I'))
        print "Predictions"
        print label_predictions
        print "Original Test set labels"
        print le.inverse_transform(labels_test)
        orig = le.inverse_transform(labels_test).flatten()
        pred = label_predictions
        return (orig, pred)
    

    def write_random_csv(self):
        pass

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

if __name__== "__main__":
    #ld = LoanData()
    #ld.train_and_predict()

    no_of_lines = 1000
    requested_file = requests.get("https://resources.lendingclub.com/LoanStats3b.csv.zip")
    zipped = zipfile.ZipFile(StringIO.StringIO(requested_file.content))
    loan_data = read_csv(zipped.open('LoanStats3b.csv'), skiprows=1, skipfooter=2)
    req_fields = loan_data[['id','loan_status']].as_matrix()
    np.random.shuffle(req_fields)
    random_lines = req_fields[:no_of_lines]
    np.savetxt('random.csv', random_lines, delimiter=',', fmt="%d,%s")
    print no_of_lines

            

