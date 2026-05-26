import math
import csv

rate = 0.0001
iterations = 1000
m = 20 #we have 20 input features here.

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inner(thetas, xs):
    total = 0
    for i in range(m+1):
        total += thetas[i] * xs[i]
    return total

def main():
    thetas = [0] * (m+1)
    for i in range(iterations):
        gradient = [0] * (m+1)
        reader = csv.DictReader(open('ancestry-train.csv'))
        for row in reader:
            xs = []
            for n in range(m+1):
                if n == 0:
                    value = 1
                else:
                    name = 'col' + str(n-1)
                    value = int(row[name])
                xs.append(value)
            y = int(row['Label'])
            for j in range(m+1):
                gradient[j] += xs[j] * (y - sigmoid(inner(thetas, xs)))
        for j in range(m+1):
            thetas[j] += rate * gradient[j]

    max_abs= 0
    max_theta = 0
    for theta in thetas:
        if abs(theta) > max_abs:
            max_abs = abs(theta)
            max_theta = theta
    print(max_theta)
    print('\n\n\n')

    for theta in thetas:
        print(theta)

if __name__ == "__main__":
    main()