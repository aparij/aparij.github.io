Title: Numpy substring search indexed result
Date: 2013-05-29 10:20
Category: Python
Tags: python, numpy
Slug: numpy-array-string-search
Author: Alex Parij
Summary: Short version for index and feeds

Working on Kaggle’s Titanic competition I needed to test each Numpy array cell if the string s1 contains the second string s2 and return an indexed array with True/False values.
Let's define an array:

    In [2]: import numpy as np
    
    In [3]: nparr = np.array([["aaMRac","bbbb"],["ccc","ffff"],["eeee","gggggg"]]
    
    In [4]: nparr
    Out[4]: 
    array([["aaMRac", "bbbb"],
           ["ccc", "ffff"],
           ["eeee", "gggggg"]], 
          dtype="|S6")

and I’m looking for strings that contain ‘MR’. I should get :

    [True, False],
    [False, False],
    [False, False]

because ‘aaMRac’ is the only cell that one contains ‘MR’.
Trying :

    In [5]: "MR" in nparr
    Out[5]: False

Gives me False because it tests for a string to string equality and returns the answer for the entire array.

To get the indexed answer I do

    In [6]: np.array(["MR" in s for s in nparr.flat]).reshape(nparr.shape)
    Out[6]: 
    array([[ True, False],
   	   [False, False],
      	   [False, False]], dtype=bool)

which flattens the array before looking for the substring using a list comprehension. It then creates the new indexed answer with the right array dimensions.
 
If you want to select only one column, you do like this:

    In [8]: np.array(["MR" in s for s in nparr[0:,1].flat])
    Out[8]: array([False, False, False], dtype=bool)

