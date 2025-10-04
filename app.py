import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact, IntRangeSlider
from IPython.display import clear_output

# Load Titanic dataset (using seaborn built-in dataset)
titanic = sns.load_dataset('titanic')

# Clean data - fill missing age with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

def print_summary(data):
    total_passengers = len(data)
    survived_count = data['survived'].sum()
    survival_rate = survived_count / total_passengers * 100 if total_passengers else 0
    average_age = data['age'].mean()
    average_age_male = data[data['sex'] == 'male']['age'].mean()
    average_age_female = data[data['sex'] == 'female']['age'].mean()
    class_counts = data['pclass'].value_counts().sort_index()
    most_common_survivor_age = data[data['survived'] == 1]['age'].mode()
    most_common_survivor_age_text = most_common_survivor_age.iloc[0] if not most_common_survivor_age.empty else "No data"
    survival_rate_male = data[data['sex'] == 'male']['survived'].mean() * 100 if total_passengers else 0
    survival_rate_female = data[data['sex'] == 'female']['survived'].mean() * 100 if total_passengers else 0
    more_likely_gender = "Female" if survival_rate_female > survival_rate_male else "Male"

    print(f"Data Summary (Filtered) - {total_passengers} passengers:")
    print(f"Survived: {survived_count} ({survival_rate:.2f}%)")
    print(f"Average Age: {average_age:.2f} years")
    print(f"Average Male Age: {average_age_male:.2f} years; Female Age: {average_age_female:.2f} years")
    print(f"Passenger Class Counts: 1st - {class_counts.get(1, 0)}, 2nd - {class_counts.get(2, 0)}, 3rd - {class_counts.get(3, 0)}")
    print(f"Most Common Survivor Age: {most_common_survivor_age_text}")
    print(f"Male Survival Rate: {survival_rate_male:.2f}% | Female Survival Rate: {survival_rate_female:.2f}%")
    print(f"Gender More Likely to Survive: {more_likely_gender}")

def plot_dashboard(data):
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 2, 1)
    data['survived'].value_counts().plot.pie(
        labels=['Not Survived', 'Survived'], autopct='%1.1f%%',
        colors=['#ff9999', '#8fd9b6'], startangle=90)
    plt.title('Survival Rate')
    plt.ylabel('')

    plt.subplot(2, 2, 2)
    sns.countplot(x='sex', hue='survived', data=data, palette='pastel')
    plt.title('Survival by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(['Not Survived', 'Survived'])

    plt.subplot(2, 2, 3)
    sns.countplot(x='pclass', hue='survived', data=data, palette='muted')
    plt.title('Survival by Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.legend(['Not Survived', 'Survived'])

    plt.subplot(2, 2, 4)
    sns.histplot(data=data, x='age', bins=25, hue='survived',
                 multiple='stack', palette='Set2')
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.boxplot(x='survived', y='age', data=data, palette='Set3')
    plt.title('Age Distribution by Survival')
    plt.xlabel('Survived')
    plt.ylabel('Age')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='pclass', y='survived', hue='sex', data=data, palette='Set1')
    plt.title('Survival Rate by Class and Gender')
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')
    plt.show()

def interactive_dashboard(age_range):
    clear_output(wait=True)
    min_age, max_age = age_range
    filtered_data = titanic[(titanic['age'] >= min_age) & (titanic['age'] <= max_age)]

    print_summary(filtered_data)
    plot_dashboard(filtered_data)

age_slider = IntRangeSlider(
    value=[0, 80], min=0, max=80, step=1,
    description='Age Range:', continuous_update=True
)

interact(interactive_dashboard, age_range=age_slider)
