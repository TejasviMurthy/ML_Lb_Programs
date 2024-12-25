from statistics import mean
import math


def stdDeviation(listt, meann, lenn):
    sqr = 0
    for i in range(len(listt)):
        sqr += (listt[i] - meann) ** 2
    variance = sqr / lenn
    stDev = math.sqrt(variance)
    return stDev


age = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
fat = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]
n = len(fat)
ageMean = mean(age)
fatMean = mean(fat)
ab = 0
for i in range(n):
    ab += (age[i] * fat[i])

ageSD = stdDeviation(age, ageMean, n)
fatSD = stdDeviation(fat, fatMean, n)

cor = (ab - (n * ageMean * fatMean)) / (n * ageSD * fatSD)

print(cor)
