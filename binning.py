from statistics import mean, median, mode
data = [4, 8, 15, 21, 21, 24, 25, 28, 34]
bins = 3
list1 = data[0:3]
list2 = data[3:6]
list3 = data[6:9]


print("\nBinning by Median")
medianList1 = [median(list1)] * len(list1)
medianList2 = [median(list2)] * len(list2)
medianList3 = [median(list3)] * len(list3)

print(f"Bin 1 (Median): {medianList1}")
print(f"Bin 2 (Median): {medianList2}")
print(f"Bin 3 (Median): {medianList3}")

print("\nBinning by Mean")
meanList1 = [mean(list1)] * len(list1)
meanList2 = [mean(list2)] * len(list2)
meanList3 = [mean(list3)] * len(list3)

print(f"Bin 1 (Mean): {meanList1}")
print(f"Bin 2 (Mean): {meanList2}")
print(f"Bin 3 (Mean): {meanList3}")


# Binning by Mode
print("\nBinning by Mode")
# For mode, if there's a tie, the function will return the first mode encountered
modeList1 = [mode(list1)] * len(list1)
modeList2 = [mode(list2)] * len(list2)
modeList3 = [mode(list3)] * len(list3)

print(f"Bin 1 (Mode): {modeList1}")
print(f"Bin 2 (Mode): {modeList2}")
print(f"Bin 3 (Mode): {modeList3}")
