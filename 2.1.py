import pandas as pd
import math

df = pd.read_excel('example.xlsx')


def calculate_mode(column):
    frequency = {}
    for value in column:
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            if value in frequency:
                frequency[value] += 1
            else:
                frequency[value] = 1
    if frequency:
        mode = max(frequency, key=frequency.get)
        return mode
    return None


def calculate_median(column):
    clean_data = []
    for value in column:
        if value is not None:
            clean_data.append(value)

    for i in range(len(clean_data)):
        for j in range(i + 1, len(clean_data)):
            if clean_data[i] > clean_data[j]:
                clean_data[i], clean_data[j] = clean_data[j], clean_data[i]

    n = len(clean_data)

    if n == 0:
        return None

    if n % 2 == 1:
        median = clean_data[n // 2]
    else:
        mid1 = clean_data[n // 2 - 1]
        mid2 = clean_data[n // 2]
        median = (mid1 + mid2) / 2

    return median


for col in df.columns:
    column_data = df[col].tolist()

    if df[col].dtype == 'object':
        mode_value = calculate_mode(column_data)
        for i in range(len(column_data)):
            if column_data[i] is None or (isinstance(column_data[i], float) and math.isnan(column_data[i])):
                column_data[i] = mode_value

        df[col] = column_data

    median_value = calculate_median(column_data)
    for i in range(len(column_data)):
        if column_data[i] is None or (isinstance(column_data[i], float) and math.isnan(column_data[i])):
            column_data[i] = median_value

    df[col] = column_data
print(df)

