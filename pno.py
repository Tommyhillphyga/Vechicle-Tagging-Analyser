import time
import random


def calculatePrimeFactors(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d) # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac

def main():
    print("Starting number crunching")
    t0 = time.time()
    for i in range(10000):
        randi = random.randint(20000, 100000000)
        calculatePrimeFactors(randi)
    t1 = time.time()
        
    totalTime = t1 - t0
    print("Execution Time: {}".format(totalTime))


if __name__ == '__main__':
    main()