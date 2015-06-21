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
from lendup import LoanData
ld = LoanData()
ld.train_and_learn()

```


