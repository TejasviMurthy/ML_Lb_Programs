min = int(input("enter min number : "))
max = int(input("enter maxx number : "))
nor_min = int(input("enter normalised min : "))
nor_max = int(input("enter normalised max : "))

num = int(input("enter number to be transformed : "))

norm = ((num-min)/(max-min)*(nor_max-nor_min)+nor_min)

print(f"by min max normalization {num} is transformed to {norm}", )