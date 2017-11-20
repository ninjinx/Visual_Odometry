n = 30
x = n
checks = 0
for i in range(n):
    checks += x

print(checks)

checks2 = 0
for i in range(n):
    checks2 += x
    x -= 1

print(checks2)

print(checks/checks2)