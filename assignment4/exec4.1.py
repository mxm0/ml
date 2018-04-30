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

plt.bar(y, age_groups)
#plt.show()

# Part b - 

def probDisease(entry):
  overall = 0
  num = 0
  overall = df[entry].count()
  num = df[entry].sum()
  return float(num/overall)

def probResidence(entry):
  overall = 0
  num = 0
  overall = df[entry].count()
  for residence in df[entry]:
    if residence == 1:
      num += 1
  return float(num/overall)

def probSib(entry):
  overall = 0
  num = 0
  overall = df[entry].count()
  for e in df[entry]:
    if e  >= 1:
      num += 1
  return float(num/overall)


print(probDisease("diseaseX"))
print(probResidence("residence"))
print(probSib("olderSiblings"))






