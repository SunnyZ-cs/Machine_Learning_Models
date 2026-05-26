from scipy import stats
import math

total = 1472

def appr_normal(arg):
    if arg == 'A':
        file = 'personKeyTimingA.txt'
    else:
        file = 'personKeyTimingB.txt'
    with open(file, 'r') as f:
        total_durations = 0
        total_durations_square = 0
        prev = 0
        for line in f:
            parts = line.strip().split(',')
            timestamp = float(parts[0])
            duration = timestamp - prev
            prev = timestamp
            total_durations += duration
            total_durations_square += duration ** 2
        mean = total_durations/total
        var = total_durations_square/total - mean ** 2
        return [mean, var]

def keystrokes():
    norm_A = appr_normal('A')
    norm_B = appr_normal('B')
    X = stats.norm(norm_A[0], math.sqrt(norm_A[1]))
    Y = stats.norm(norm_B[0], math.sqrt(norm_B[1]))

    ratio = 1
    prev = 0
    with open('email.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            timestamp = float(parts[0])
            duration = timestamp - prev
            prev = timestamp
            ratio *= X.pdf(duration)/Y.pdf(duration)
    print(ratio)

def main():
    keystrokes()


if __name__ == '__main__':
    main()