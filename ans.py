from math import factorial

N = []
digits = list(range(10))


count = 0

for loop in range(10):
    for _, digit in enumerate(digits):
        count += factorial(len(digits) - 1)
        if count < 1000000:
            continue
        else:
            count -= factorial(len(digits) - 1)
            break

    N.append(digit)
    digits.remove(digit)

print(N)