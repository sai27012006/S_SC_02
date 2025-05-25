import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1. Create and Save CSV Dataset
# -------------------------------
data = {
    "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Survived":    [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    "Pclass":      [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
    "Name": [
        "Braund, Mr. Owen Harris",
        "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "Heikkinen, Miss. Laina",
        "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
        "Allen, Mr. William Henry",
        "Moran, Mr. James",
        "McCarthy, Mr. Timothy J",
        "Palsson, Master. Gosta Leonard",
        "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",
        "Nasser, Mrs. Nicholas (Adele Achem)"
    ],
    "Sex":        ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
    "Age":        [22, 38, 26, 35, 35, None, 54, 2, 27, 14],
    "SibSp":      [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
    "Parch":      [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    "Ticket":     ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450", "330877", "17463", "349909", "347742", "237736"],
    "Fare":       [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708],
    "Embarked":   ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"]
}
df = pd.DataFrame(data)
df.to_csv("titanic.csv", index=False)
print("✅ Sample Titanic dataset saved as 'titanic.csv'.")

# ---------------------------------
# 2. Load and Clean the Data
# ---------------------------------
df = pd.read_csv('titanic.csv')

# Drop columns not needed for analysis
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Convert 'Sex' to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# ---------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------
sns.set(style='whitegrid')

# Plot survival count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Survival by gender
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.xticks([0, 1], ['Male', 'Female'])
plt.show()

# Survival by passenger class
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.show()

# ---------------------------------
# 4. Summary Output
# ---------------------------------
print("\n📊 Summary: Mean Survival Rates by Sex and Pclass")
print(df[['Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass']).mean())
