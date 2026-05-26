import math
import csv
rate = 0.000001
iterations = 1000
m = 22 #we have 22 input features here.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def inner(thetas, xs):
    total = 0
    for i in range(m+1):
        total += thetas[i] * xs[i]
    return total
def predict(thetas, xs):
    prob = sigmoid(inner(thetas, xs))
    return 1 if prob >= 0.5 else 0
def test(thetas):
    n_correct = 0
    reader = csv.DictReader(open('heart-test.csv'))
    for row in reader:
        xs = [1, int(row['A.1']), int(row['A.2']), int(row['A.3']), int(row['A.4']),
              int(row['B.1']), int(row['B.2']), int(row['B.3']), int(row['B.4']), int(row['B.5']),
              int(row['C.1']), int(row['C.2']), int(row['C.3']), int(row['C.4']), int(row['C.5']),
              int(row['D.1']), int(row['D.2']), int(row['D.3']), int(row['D.4']),
              int(row['E.1']), int(row['E.2']), int(row['E.3']), int(row['E.4'])
              ]
        actual_y = int(row['Label'])
        predict_y = predict(thetas, xs)
        if predict_y == actual_y:
            n_correct += 1
    print(f'\nAccuracy: {n_correct / 187}')
def main():
    thetas = [0] * (m+1)
    for i in range(iterations):
        gradient = [0] * (m+1)
        reader = csv.DictReader(open('heart-train.csv'))
        for row in reader:
            xs = [1, int(row['A.1']),int(row['A.2']), int(row['A.3']), int(row['A.4']),
                  int(row['B.1']), int(row['B.2']), int(row['B.3']), int(row['B.4']), int(row['B.5']),
                  int(row['C.1']), int(row['C.2']), int(row['C.3']), int(row['C.4']), int(row['C.5']),
                  int(row['D.1']), int(row['D.2']), int(row['D.3']), int(row['D.4']),
                  int(row['E.1']), int(row['E.2']), int(row['E.3']), int(row['E.4'])
                  ]
            y = int(row['Label'])
            for j in range(m+1):
                gradient[j] += xs[j] * (y - sigmoid(inner(thetas, xs)))
        for j in range(m+1):
            thetas[j] += rate * gradient[j]
    test(thetas)

if __name__ == "__main__":
    main()