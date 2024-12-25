import pandas as pd

data = pd.read_excel('bn_data.xlsx')
df = pd.DataFrame(data)

total_count = len(df)
prior_yes = len(df[df["Class_label"] == "yes"]) / total_count
prior_no = len(df[df["Class_label"] == "no"]) / total_count


def conditional_probability(attribute, value, class_label):
    subset = df[df["Class_label"] == class_label]
    return len(subset[subset[attribute] == value]) / len(subset)

X = {"Age": "youth", "Income": "medium", "Student": "yes", "Credit_rating": "fair"}

p_x_given_yes = (
    conditional_probability("Age", X["Age"], "yes") *
    conditional_probability("Income", X["Income"], "yes") *
    conditional_probability("Student", X["Student"], "yes") *
    conditional_probability("Credit_rating", X["Credit_rating"], "yes")
)

p_x_given_no = (
    conditional_probability("Age", X["Age"], "no") *
    conditional_probability("Income", X["Income"], "no") *
    conditional_probability("Student", X["Student"], "no") *
    conditional_probability("Credit_rating", X["Credit_rating"], "no")
)

posterior_yes = p_x_given_yes * prior_yes
posterior_no = p_x_given_no * prior_no

if (posterior_yes> posterior_no):
	prediction = "yes"
else:
	prediction = "no"
print(f"P(X | buys computer = yes): {p_x_given_yes:.5f}")
print(f"P(X | buys computer = no): {p_x_given_no:.5f}")
print(f"P(X) * P(buys computer = yes): {posterior_yes:.5f}")
print(f"P(X) * P(buys computer = no): {posterior_no:.5f}")
print(f"Prediction: buys computer = {prediction}")
