import math
n = 7
total = 0
f = int(math.factorial(n)/(math.factorial(3)*math.factorial(n-3)))

total = 0

for i in range(f):
    m = math.factorial(f)/(math.factorial(i+1)*math.factorial(f-i-1))
    total = total + m*n*(1 + 2**(n-2))

n = 5
f = int(math.factorial(n)/(math.factorial(3)*math.factorial(n-3)))

total2 = 0

for i in range(f):
    m = math.factorial(f)/(math.factorial(i+1)*math.factorial(f-i-1))
    total2 = total2 + m*n*(1 + 2**(n-2))

print(total/total2)



