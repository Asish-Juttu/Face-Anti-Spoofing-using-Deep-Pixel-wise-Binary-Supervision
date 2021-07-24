import pandas as pd
import os
import csv

#csv = pd.read_csv('data/nuaa/lol.csv')

path = 'data/nuaa/real_test'
root1 = 'real_test'
path2 = 'data/nuaa/fake_test'
root2 = 'fake_test'

with open('data/nuaa/lol2.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  #writer.writerow(['identifier', 'file', 'description', 'subject[0]', 'title', 'creator', 'date', 'collection'])
  for root, dirs, files in os.walk(path):
    for filename in files:
        writer.writerow([os.path.join(root1,filename),1])
  for root, dirs, files in os.walk(path2):
      for filename in files:
          writer.writerow([os.path.join(root2,filename),0])