n = 7
import math

total = 0
f = int(math.factorial(n)/(math.factorial(3)*math.factorial(n-3)))
m = math.factorial(n)/(math.factorial(n-2)*2)

for i in range(f):
    total = total + math.factorial(f)/(math.factorial(i+1)*math.factorial(f-i-1))

total = m*total

print(total)