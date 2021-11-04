import random

arr = []

for i in range(30):
    arr.append(str(round(random.normalvariate(450, 10))))

print("\t".join(arr))