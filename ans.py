import collections

def factors(n):
    res, prim = collections.defaultdict(int), 2;
    while n > 1:
        if (n % prim == 0):
            n //= prim
            res[prim] += 1
        else:
            prim += 1
    return res

size = 10000010;
num = 0

while (num < size):
    if len(factors(num+3))<4: num+=4
    elif len(factors(num+2))<4: num+=3
    elif len(factors(num+1))<4: num+=2
    elif len(factors(num))<4: num+=1
    else:
        print(num,factors(num))
        print(num+1,factors(num+1))
        print(num+2,factors(num+2))
        print(num+3,factors(num+3))
        break