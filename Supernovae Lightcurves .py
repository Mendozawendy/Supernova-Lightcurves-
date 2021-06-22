#!/usr/bin/env python
# coding: utf-8

# In[2]:


3 + 5 * 4


# In[4]:


weight_kg = 60.0


# In[5]:


weight_kg_text = 'weight in kilograms:'


# In[6]:


print(weight_kg)


# In[7]:


print(weight_kg_text, weight_kg)


# In[8]:


print('weight in pounds:', 2.2 * weight_kg)


# In[9]:


print(weight_kg)


# In[10]:


weight_kg = 65.0
print('weight in kilogram is now:', weight_kg)


# In[11]:


# There are 2.2 punds per kilogram 
weight_lb = 2.2 * weight_kg
print(weight_kg_text, weight_kg, 'and in pounds:', weight_lb)


# In[12]:


weight_kg = 100 
print('weight in kilogram is now:', weight_kg, 'and weight in pounds still:', weight_lb)


# In[13]:


import numpy 


# In[19]:


numpy.loadtxt(fname='03D1ar.csv', delimiter=',', skiprows=1)


# In[20]:


data = numpy.loadtxt(fname='03D1ar.csv', delimiter=',', skiprows=1)


# In[21]:


print(data)


# In[22]:


print(type(data))


# In[23]:


print(data.dtype)


# In[24]:


print(data.shape)


# In[26]:


print('first value in data:', data[0, 0])


# In[27]:


print('middle value in data:', data[24,5])


# In[29]:


print(data[0:4, 0:9])


# In[30]:


print(data[5:10, 0:9])


# In[31]:


small = data[:3, 7:] 
print('small is:')
print(small)


# In[32]:


doubledata = data * 2.0


# In[33]:


print('original')
print(data[:3, 7:])
print('doubledata:')
print(doubledata[:3, 7:])


# In[38]:


tripledata = doubledata + data


# In[39]:


print('tripledata:')
print(tripledata[:3, 7:])


# In[41]:


print(numpy.mean(data[:,1]))


# In[42]:


print(numpy.nanmean(data[:,1]))


# In[43]:


import time 
print(time.ctime())


# In[51]:


maxval, minval, stdval = numpy.nanmax(data[:,1]), numpy.nanmin(data[:,1]), numpy.nanstd(data[:,1])

print('maximum flux:', maxval)
print('minimum flux:', minval)
print('standard deviation:', stdval)


# In[53]:


Flux_r = data[:, 3] # everything on the first axis (rows), the fourth on the second axis (columnd) 
print('maximum flux in g band is:', numpy.nanmax(Flux_r))


# In[56]:


print('maximum flux in i band is:', numpy.nanmax(data[:,5]))


# In[58]:


print(numpy.nanmean(data, axis=0))


# In[59]:


print(numpy.nanmean(data, axis=0).shape)


# In[60]:


print(numpy.nanmean(data,axis=1))


# In[61]:


index = data[:1] > 100
print(index)


# In[62]:


numpy.sum(index)


# In[64]:


print(data[:,1][index])


# In[65]:


numpy.zeros(5)


# In[66]:


mass = 47.5 
age = 122
mass = mass * 2.0
age = age - 20 
print(mass, age)


# In[68]:


first, second ='Grace', 'Hopper'
third, fourth = second, first 
print(third, fourth)


# In[73]:


element = 'oxygen'
print('first three characters:', element[0:3])
print('last three characters:', element[3:6])


# In[77]:


import matplotlib


# In[84]:


import numpy

A = numpy.array([[1,2,3], [4,5,6], [7, 8, 9]])
print('A = ')
print(A)

B = numpy.hstack([A, A])
print('B = ')
print(B)

C = numpy.vstack([A, A])
print('C = ')
print(C)


# In[85]:


import matplotlib.pyplot 
image = matplotlib.pyplot.imshow(data[:,1:])
matplotlib.pyplot.show()


# In[86]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[89]:


ave_flux = numpy.nanmean(data[:,1:], axis=1)
mjd = data[:,0]
ave_plot = matplotlib.pyplot.plot(mjd, ave_flux)
matplotlib.pyplot.show()


# In[90]:


max_plot = matplotlib.pyplot.plot(mjd, numpy.nanmax(data[:,1:], axis=1))
matplotlib.byplot.show()


# In[91]:


min_plot = matplotlib.pyplot.plot(mjd,numpy.nanmin(data[:,1:], axis=1))
matplotlib.pyplot.show()


# In[92]:


matplotlib.pyplot.plot(mjd,data[:,1],'o', color='blue')
matplotlib.pyplot.plot(mjd,data[:,3],'o', color='green')
matplotlib.pyplot.plot(mjd,data[:,5],'o', color='yellow')
matplotlib.pyplot.plot(mjd,data[:,7],'o', color='red')
matplotlib.pyplot.show()


# In[93]:


import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname='03D1ar.csv', delimiter=',', skiprows=1)
mjd = data[:,0]

fig = plt.figure(figsize=(15.0, 4.0))

axes1 = fig.add_subplot(1, 4, 1)
axes2 = fig.add_subplot(1, 4, 2)
axes3 = fig.add_subplot(1, 4, 3)
axes4 = fig.add_subplot(1, 4, 4)

axes1.set_xlabel('MJD')
axes1.set_ylabel('g')
axes1.set_ylim([-150,720])
axes1.plot(mjd,data[:,1],'o', color='blue')

axes2.set_xlabel('MJD')
axes2.set_ylabel('r')
axes2.set_ylim([-150,720])
axes2.plot(mjd,data[:,3],'o', color='green')

axes3.set_xlabel('MJD')
axes3.set_ylabel('i')
axes3.set_ylim([-150,720])
axes3.plot(mjd,data[:,5],'o', color='yellow')

axes4.set_xlabel('MJD')
axes4.set_ylabel('z')
axes4.set_ylim([-150,720])
axes4.plot(mjd, data[:,7],'o', color='red')

fig.tight_layout()

plt.show(block=False)


# In[94]:


word = 'lead'


# In[96]:


print(word[0])
print(word[1])
print(word[2])
print(word[3])


# In[98]:


word = 'oxygen'
for char in word:
    print(char)


# In[107]:


word = 'oxygen'
for banana in word: 
    print(banana)


# In[108]:


length = 0 
for vowel in 'aeiou':
    length = length + 1
    print('There are', length, 'vowels')


# In[109]:


letter = 'z'
for letter in 'abc':
    print(letter)
print('after the loop, letter is', letter)


# In[110]:


print(len('aeiou'))


# In[111]:


for i in range(1,4): 
    print(i)


# In[112]:


print(5 ** 3)


# In[113]:


result = 1 
for i in range(0,3):
    result = result * 5
    print(result)


# In[114]:


newstring = ''
oldstring = 'Newton'
for char in oldstring: 
    newstring = char + newstring 
    print(newstring)


# In[115]:


odds = [1, 3, 5, 7]
print('odds are:', odds)


# In[116]:


print('first and last:', odds[0], odds[-1])


# In[117]:


for number in odds: 
    print(number)


# In[118]:


names = ['Curie', 'Darwing', 'Turing']  # typo in Darwin's name
print('names is originally:', names)
names[1] = 'Darwin'  # correct the name
print('final value of names:', names)


# In[119]:


salsa = ['peppers', 'onions', 'cilantro', 'tomatoes']
my_salsa = salsa        # <-- my_salsa and salsa point to the *same* list data in memory
salsa[0] = 'hot peppers'
print('Ingredients in my salsa:', my_salsa)


# In[120]:


salsa = ['peppers', 'onions', 'cilantro', 'tomatoes']
my_salsa = list(salsa)        # <-- makes a *copy* of the list
salsa[0] = 'hot peppers'
print('Ingredients in my salsa:', my_salsa)


# In[121]:


x = [['pepper', 'zucchini', 'onion'],
     ['cabbage', 'lettuce', 'garlic'],
     ['apple', 'pear', 'banana']]


# In[122]:


print([x[0]])


# In[123]:


print(x[0])


# In[124]:


'pepper'


# In[125]:


odds.append(11)
print('odds after adding a value:', odds)


# In[126]:


del odds[0]
print('odds after removing the first element:', odds)


# In[127]:


odds.reverse()
print('odds after reversing:', odds)


# In[128]:


odds = [1, 3, 5, 7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)


# In[129]:


my_list = []
for char in "hello":
    my_list.append(char)
print(my_list)


# In[130]:


import glob 


# In[132]:


print(glob.glob('*.csv'))


# In[134]:


import numpy 
import glob 

filenames = sorted(glob.glob('*.csv')) 
filenames = filenames[0:3]
for f in filenames: 
    print(f) 
    
    data = numpy.loadtxt(fname=f, delimiter=',', skiprows=1)
    
    y_min = numpy.nanmin(data[:,[1,3,5,7]])
    y_max = numpy.nanmax(data[:,[1,3,5,7]])
    
    print(f, 'Min brightness: ', y_min, 'Max brightness: ', y_max)


# In[135]:


import glob
import numpy
import matplotlib.pyplot

filenames = glob.glob('inflammation*.csv')
composite_data = numpy.zeros((60,40))

for f in filenames:
    data = numpy.loadtxt(fname = f, delimiter=',')
    composite_data += data

composite_data/=len(filenames)

fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))

axes1 = fig.add_subplot(1, 3, 1)
axes2 = fig.add_subplot(1, 3, 2)
axes3 = fig.add_subplot(1, 3, 3)

axes1.set_ylabel('average')
axes1.plot(numpy.mean(composite_data, axis=0))

axes2.set_ylabel('max')
axes2.plot(numpy.max(composite_data, axis=0))

axes3.set_ylabel('min')
axes3.plot(numpy.min(composite_data, axis=0))

fig.tight_layout()

matplotlib.pyplot.show()


# In[136]:


num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')


# In[137]:


num = 53
print('before conditional...')
if num > 100:
    print(num,' is greater than 100')
print('...after conditional')


# In[138]:


num = -3

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')


# In[139]:


if (1 > 0) and (-1 > 0):
    print('both parts are true')
else:
    print('at least one part is false')
    


# In[140]:


if (1 < 0) or (-1 < 0):
    print('at least one test is true')


# In[141]:


import numpy as np


# In[142]:


if np.nanmin(data[:,1]) < 0:
    print('Negative fluxes!')


# In[145]:


if np.sum(np.isnan(data[:,1])) == data.shape[0]:
    print(f + ': A NaN column')


# In[146]:





# In[147]:


f1 = '03D3af.csv'
data = np.loadtxt(fname=f1, delimiter=',', skiprows=1)


if np.nanmin(data[:,1]) < 0.:
    print(f1 + ': a negative flux')
elif np.sum(np.isnan(data[:,1])) == data.shape[0]:
    print(f1 + ': a NaN column')
else:
    print('Seems OK!')


# In[151]:


f2 = '03D1ar.csv'
data = np.loadtxt(fname=f2, delimiter=',')

if np.nanmin(data[:,1]) < 0.:
    print(f + ': a negative flux')
elif np.sum(np.isnan(data[:,1])) == data.shape[0]:
    print(f + ': a NaN column')
else:
    print('Seems OK!')


# In[152]:


for file in files:
    if file.startswith('inflammation-'):
        large_files.append(file)
    elif file.startswith('small-'):
        small_files.append(file)
    else:
        other_files.append(file)

print('large_files:', large_files)
print('small_files:', small_files)
print('other_files:', other_files)


# In[153]:


files = ['inflammation-01.csv',
         'myscript.py',
         'inflammation-02.csv',
         'small-01.csv',
         'small-02.csv']
large_files = []
small_files = []
other_files = []


# In[154]:


large_files = ['inflammation-01.csv', 'inflammation-02.csv']
small_files = ['small-01.csv', 'small-02.csv']
other_files = ['myscript.py']


# In[155]:


for file in files:
    if file.startswith('inflammation-'):
        large_files.append(file)
    elif file.startswith('small-'):
        small_files.append(file)
    else:
        other_files.append(file)

print('large_files:', large_files)
print('small_files:', small_files)
print('other_files:', other_files)


# In[156]:


def fahr_to_celsius(temp):
    return ((temp - 32) * (5/9))


# In[157]:


fahr_to_celsius(32)


# In[158]:


print('freezing point of water:', fahr_to_celsius(32), 'C')
print('boiling point of water:', fahr_to_celsius(212), 'C')


# In[159]:


def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

print('freezing point of water in Kelvin:', celsius_to_kelvin(0.))


# In[160]:


def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print('boiling point of water in Kelvin:', fahr_to_kelvin(212.0))


# In[161]:


def detect_problems(filename):

    data = np.loadtxt(fname=filename, delimiter=',', skiprows=1)

    for i in [1,3,5,7]:
        if np.nanmin(data[:,i]) < 0.:
            print(filename + ': A negative flux.')
        elif np.sum(np.isnan(data[:,i])) == data.shape[0]:
            print(filename + ': A NaN column')
        else:
            print('Seems OK!')


# In[162]:


import glob
import numpy as np
filenames = sorted(glob.glob('*.csv'))

for f in filenames[:3]:
    print(f)
    detect_problems(f)


# In[163]:


#fahr_to_celsius(temp):
#converts fahrenheit degrees to celsius degrees
def fahr_to_celsius(temp):
    return ((temp - 32) * (5/9))


# In[164]:


def fahr_to_celsius(temp):
    """Converts fahrenheit degrees to celsius degrees."""
    return ((temp - 32) * (5/9))


# In[165]:


help(fahr_to_celsius)


# In[166]:


def fahr_to_celsius(temp):
    """
    Converts fahrenheit degrees to celsius degrees.
    Example:
    >fahr_to_celsius(125)
    37.77777777777778
    """
    return ((temp - 32) * (5/9))

help(fahr_to_celsius)


# In[168]:


np.loadtxt('03D1ar.csv', delimiter=',',skiprows=1)


# In[169]:


def s(p):
    a = 0
    for v in p:
        a += v
    m = a / len(p)
    d = 0
    for v in p:
        d += (v - m) * (v - m)
    return np.sqrt(d / (len(p) - 1))

def std_dev(sample):
    sample_sum = 0
    for value in sample:
        sample_sum += value

    sample_mean = sample_sum / len(sample)

    sum_squared_devs = 0
    for value in sample:
        sum_squared_devs += (value - sample_mean) * (value - sample_mean)

    return np.sqrt(sum_squared_devs / (len(sample) - 1))


# In[170]:


print(fence('name', '*'))


# In[171]:


def fence(original, wrapper):
    return wrapper + original + wrapper


# In[172]:


def rescale(input_array):
    L = np.min(input_array)
    H = np.max(input_array)
    output_array = (input_array - L) / (H - L)
    return output_array


# In[173]:


# This code has an intentional error. You can type it directly or
# use it for reference to understand the error message below.
def favorite_ice_cream():
    ice_creams = [
        "chocolate",
        "vanilla",
        "strawberry"
    ]
    print(ice_creams[3])

favorite_ice_cream()


# In[175]:


def some_function():
    msg = "hello, world!"
    print(msg)
     return msg


# In[176]:


def some_function():
	msg = "hello, world!"
	print(msg)
        return msg


# In[177]:


print(a)


# In[178]:


print(hello)


# In[179]:


for number in range(10):
    count = count + number
print("The count is:", count)


# In[180]:


Count = 0
for number in range(10):
    count = count + number
print("The count is:", count)


# In[181]:


letters = ['a', 'b', 'c']
print("Letter #1 is", letters[0])
print("Letter #2 is", letters[1])
print("Letter #3 is", letters[2])
print("Letter #4 is", letters[3])


# In[182]:


file_handle = open('myfile.txt', 'r')


# In[183]:


file_handle = open('myfile.txt', 'w')
file_handle.read()


# In[189]:


def another_function():
    print("Syntax errors are annoying.")
    print("But at least Python tells us about them!")
    print("So they are usually not too hard to fix.")


# In[193]:


message = ""
for number in range(10):
    # use a if the number is a multiple of 3, otherwise use b
    if (number % 3) == 0:
        message = message + "a"
    else:
        message = message + "b"
print(message)


# In[195]:


seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print('My favorite season is ', seasons[-1])


# In[196]:


python ../code/readings_04.py --mean inflammation-01.csv


# In[ ]:




