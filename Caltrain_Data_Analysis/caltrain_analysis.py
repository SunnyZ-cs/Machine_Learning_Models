import csv
import math
from sklearn.linear_model import LinearRegression
from scipy import stats

def LR():
    model = LinearRegression()
    y_lst = []
    x_lst = []
    reader = csv.DictReader(open('caltrain-train.csv'))
    for row in reader:
        Xi= [float(row['is_summer']), float(row['is_weekend']),
             float(row['is_holiday']), float(row['busy_time_of_day']),
             float(row['north_or_southbound']), float(row['temperature']),
             float(row['chance_of_rain'])]
        x_lst.append(Xi)
        y_lst.append(float(row['passengers_per_hour']))
    model.fit(x_lst, y_lst)

    test_x_lst = []
    test_y_lst = []
    reader1 = csv.DictReader(open('caltrain-test.csv'))
    for row in reader1:
        test_Xi = [float(row['is_summer']), float(row['is_weekend']),
             float(row['is_holiday']), float(row['busy_time_of_day']),
             float(row['north_or_southbound']), float(row['temperature']),
             float(row['chance_of_rain'])]
        test_x_lst.append(test_Xi)
        test_y_lst.append(float(row['passengers_per_hour']))

    pred_y = model.predict(test_x_lst)

    sum = 0
    for i in range(len(pred_y)):
        sum += (pred_y[i] - test_y_lst[i]) ** 2
    var = sum/len(pred_y)
    print(var)

    X = stats.norm(0, math.sqrt(var))
    prob = (1-X.cdf(20))*2
    print(prob)


def main():
    LR()

if __name__ == "__main__":
    main()
