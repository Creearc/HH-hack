import csv

arr = ''

csvfile = open('train_00.csv', 'r', encoding='utf-8')
f = csv.reader(csvfile, delimiter=',')
mx = 0
tmp = ''
i = 0
ind = 0
for s in f:
  l = len(s[0].split())
  if l > mx:
    mx = l
    tmp = s[0]
    ind = i
  i += 1
#f.close()

print(mx)
print(tmp)
print(ind)
