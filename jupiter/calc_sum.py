import csv

arr = ''

csvfile = open('train_00.csv', 'r', encoding='utf-8')
f = csv.reader(csvfile, delimiter=',')
mx = 0
tmp = ''
for s in f:
  l = len(s[0].split())
  if l > mx:
    mx = l
    tmp = s[0] 
#f.close()

print(mx)
print(tmp)
