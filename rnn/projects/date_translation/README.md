# date translation 
date translation from wide range of formats to a universal format 'yyy-mm-dd'
the model is basically a bidirectional LSTM with attention.

# examples on date machine translation
- original 27 jan 2010 --> 2010-01-27
- original 3 May 1979 --> 1979-05-03
- original 1 March 2001 --> 2001-03-01

# execution output
```python
padded source: [[ 5 10  0 22 13 25  0  5  3  4  3 36 36 36 36 36 36 36 36 36 36 36 36 36
  36 36 36 36 36 36]]
source to categorical: [[[0. 0. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]
  ...
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]]]
'original 27 jan 2010 translation 2010-01-27'
line: 3 May 1979
padded source: [[ 6  0 24 13 34  0  4 12 10 12 36 36 36 36 36 36 36 36 36 36 36 36 36 36
  36 36 36 36 36 36]]
source to categorical: [[[0. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  ...
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]]]
'original 3 May 1979 translation 1979-05-03'
line: 5 April 09
padded source: [[ 8  0 13 27 28 21 23  0  3 12 36 36 36 36 36 36 36 36 36 36 36 36 36 36
  36 36 36 36 36 36]]
  ...
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]]]
'original 1 March 2001 translation 2001-03-01'
````