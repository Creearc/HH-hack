import csv

arr = ''

csvfile = open('jupiter/HeadHunter_train.csv', 'r', encoding='utf-8')
f = csv.reader(csvfile, delimiter=',')
for s in f:
  print(s[3])
f.close()
