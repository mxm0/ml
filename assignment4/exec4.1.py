import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part a - Read csv, determine number of boys and girls, age group and older siblings
df = pd.read_csv("data/vaccination.csv")

count_b = 0
count_g = 0
old_sib = df["olderSiblings"].sum()
age_groups = [0,0,0,0,0]

for gender in df['gender']:
  if gender == 0:
    count_g += 1
  else:
    count_b += 1

for age in df['age']:
  age_groups[age-1] += 1

y = ["age group 1", "age group 2", "age group 3", "age group 4", "age group 5", "Boys", "Girls", "Older Siblings"]
print("Number of boys:", count_b, "Number of girls:", count_g)
age_groups.append(count_b)
age_groups.append(count_g)
age_groups.append(old_sib)

#plt.bar(y, age_groups)
#plt.show()

# Part b - 

def probDisease(entry):
  overall = 0
  num = 0
  overall = df[entry].count()
  num = df[entry].sum()
  return float(num)/overall

def probResidence(entry):
  overall = 0
  num = 0
  overall = df[entry].count()
  for residence in df[entry]:
    if residence == 1:
      num += 1
  return float(num)/overall

def probSib(entry):
  overall = 0
  num = 0
  overall = df[entry].count()
  for e in df[entry]:
    if e  >= 1:
      num += 1
  return float(num)/overall


print(probDisease("diseaseX"))
print(probResidence("residence"))
print(probSib("olderSiblings"))


# Part c
def tallerThan1M():
  d = df["height"]
  overall = d.count()
  num = 0
  for i in d:
    if i > 100.00: num += 1
  return float(num)/overall

def heavier40():
  d = df["weight"]
  overall = d.count()
  num = 0
  for i in d:
    if i > 40.00: num += 1
  return float(num)/overall
print(tallerThan1M())
print(heavier40())


def disaseYZ():
  d = df["diseaseY"]
  overall = d.count()
  num = 0
  for i in range(overall):
    if(d[i] == 1 and (df["diseaseZ"][i] == 1)): num += 1
  return float(num)/overall

print(disaseYZ())


#d
def calc(a,b,val): #(diseaseX | vacX = 0/1)
  overall = 0
  num = 0
  for i in range(df[b].count()):
    if df[b][i] == val:
      overall += 1
      if df[a][i] == 1: num += 1
  return float(num) / overall

'''
print(calc("diseaseX", "vacX",0), calc("diseaseX", "vacX",1))
print(calc("vacX", "diseaseX", 0), calc("vacX", "diseaseX", 1))

# so HERE WE SEE THAT WHEN YOU GOT X u probably have not been in place X

print(calc("diseaseY", "age",1), calc("diseaseY", "age",2), calc("diseaseY", "age",3), calc("diseaseY", "age",4))
print(calc("vacX", "age", 1), calc("vacX", "age", 2), calc("vacX", "age", 3), calc("vacX", "age", 4))
print(calc("knowsToRideABike", "vacX",0), calc("knowsToRideABike", "vacX",1))

vals = [calc("diseaseY", "age",1), calc("diseaseY", "age",2), calc("diseaseY", "age",3), calc("diseaseY", "age",4)]

fig = plt.figure()
ax = plt.axes()

x = [1,2,3,4]
ax.plot(x, vals);
vals = [calc("vacX", "age", 1), calc("vacX", "age", 2), calc("vacX", "age", 3), calc("vacX", "age", 4)]
ax.plot(x, vals);
plt.show()

# The more the age rises the more the correlation starte to rise for diseaseY
#and X is more visited the more the age rises

plt.show()
'''

#e
def calc2(b,val): #(diseaseX | vacX = 0/1)
  overall = 0
  num = 0
  for i in range(df[b].count()):
    if df[b][i] == val:
      overall += 1
      if df["diseaseY"][i] == 1 and df["diseaseZ"][i] == 1: num += 1
  return float(num) / overall

def calc3(b,val, val2): #(diseaseX | vacX = 0/1)
  overall = 0
  num = 0
  for i in range(df[b].count()):
    if df[b][i] == val and df["age"][i] == val2:
      overall += 1
      if df["diseaseY"][i] == 1 and df["diseaseZ"][i] == 1: num += 1
  return float(num) / overall

print("e) :")
print(calc2( "vacX",0), calc2( "vacX",1))
print("disX | vaxX:")
print(calc("diseaseX", "vacX",0), calc("diseaseX", "vacX",1))

print(calc3("vacX",0,1), calc3( "vacX",1,1), calc3("vacX",0,2), calc3( "vacX",1,2), calc3("vacX",0,3), calc3( "vacX",1,3), calc3("vacX",0,4), calc3( "vacX",1,4))
#by getting a new group of people there are anymore more illnesses in vacX
#----------------------------------

vals = [calc("diseaseY", "age",1), calc("diseaseY", "age",2), calc("diseaseY", "age",3), calc("diseaseY", "age",4)]

fig = plt.figure()
ax = plt.axes()

x = [1,2,3,4]
#ax.plot(x, vals);
vals = [calc3("vacX",0,1), calc3("vacX",0,2), calc3("vacX",0,3), calc3("vacX",0,4)]
ax.plot(x, vals);
vals = [calc3("vacX",1,1), calc3("vacX",1,2), calc3("vacX",1,3), calc3("vacX",1,4)]
ax.plot(x, vals);
plt.show()

# first one is higher at higher ages (more correlation)















