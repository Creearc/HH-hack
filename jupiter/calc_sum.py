import csv

arr = ''

csvfile = open('train_00.csv', 'r', encoding='utf-8')
f = csv.reader(csvfile, delimiter=',')
mx = 0
for s in f:
  l = len(s[0].split())
  if l > mx:
    mx = l
#f.close()

print(mx)
