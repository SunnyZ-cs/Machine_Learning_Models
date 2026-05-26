import math
import csv

rate = 0.0001
iterations = 1000
m = 2 #we have two input features here.

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inner(thetas, xs):
    total = 0
    for i in range(len(thetas)):
        total += thetas[i] * xs[i]
    return total

def main():
    thetas = [0] * (m+1)
    for i in range(iterations):
        gradient = [0] * (m+1)
        reader = csv.DictReader(open('simple-train.csv'))
        for row in reader:
            xs = [1, int(row['x1']), int(row['x2'])] #always assume that x0 = 1
            y = int(row['Label'])
            for j in range(len(thetas)):
                gradient[j] += xs[j] * (y - sigmoid(inner(thetas, xs)))
        for j in range(len(thetas)):
            thetas[j] += rate * gradient[j]

    print(thetas[1])

if __name__ == "__main__":
    main()