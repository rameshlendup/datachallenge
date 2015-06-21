# LendUp Datachallenge

## Requirements
python 2.7+

## Installation instructions


### Prereqs
#### OSX installation

```
sudo easy_install virtualenv
sudo easy_install pip
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

#### *nix installation

```
sudo apt-get install virtualenv
sudo apt-get install pip
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```




## Answers

### Question 1

```
(env)Rameshs-MBP:datachallenge rameshthulasiram$ python
>>> from lendup import LoanData
>>> ld = LoanData()
>>> ld.summary_statistics('loan_amnt')
count            188123.000000
mean              14354.545962
std                8115.066458
min                1000.000000
25%                8000.000000
50%               12175.000000
75%               20000.000000
max               35000.000000
count_missing    155626.000000
Name: loan_amnt, dtype: float64
>>> ld.summary_statistics('loan_status')
count                                                       188123
unique                                                           7
top                                                        Current
freq                                                        101453
unique_values    [Charged Off, Current, Default, Fully Paid, In...
value_counts     {u'Charged Off': 17406, u'Default': 72, u'Full...
Name: loan_status, dtype: object
>>> ld.summary_statistics('recoveries')
count            188123.000000
mean                 69.768592
std                 447.449357
min                   0.000000
25%                   0.000000
50%                   0.000000
75%                   0.000000
max               33520.270000
count_missing    155626.000000
Name: recoveries, dtype: float64

>>> ld.save_histogram('loan_amnt', 'loan_amnt.png')
>>>
```
![Image of Loan Amount]
(https://raw.githubusercontent.com/rameshlendup/datachallenge/master/loan_amnt.png)

### Question 2


```
>>> from lendup import LoanData
>>> ld = LoanData()
>>> orig, pred = ld.train_and_predict()
>>> confusion_matrix( orig, pred)
#########################PREDICTIONS
array([[ 4123,     1,     0,   153,     0,     0,     0],  # Charged off
       [    0, 25335,     0,     4,     0,     0,     2],  # Current
#O     [    0,    17,     0,     1,     0,     0,     0],  # Default
#R     [   13,     0,     0, 15980,     0,     0,     0],  # Fully paid
#I     [    0,   388,     0,     0,     0,     0,     0],  # In Grace Period
#G     [    0,   125,     0,     0,     0,     0,     2],  # Late (16-30 days)'
       [    0,   645,     0,     0,     0,     0,     6]]) # Late (31-120 days)
```


