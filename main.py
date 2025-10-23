"""Education Level vs Employment Rate Study
Author(s): Rasika and Subiksha
Simple analysis + a basic prediction model (linear regression)
Run: python main.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style='whitegrid', rc={{'figure.figsize':(8,5)}})

PROJECT_ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'education_employment.csv')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def basic_info(df):
    print('--- Head ---')
    print(df.head())
    print('\n--- Describe ---')
    print(df.describe())

def plot_bar_by_education(df):
    order = ['Primary', 'Secondary', 'Tertiary']
    ax = sns.barplot(data=df, x='Education_Level', y='Employment_Rate', estimator=lambda x: x.mean(), ci=None, order=order)
    ax.set_title('Average Employment Rate by Education Level')
    plt.savefig(os.path.join(IMAGES_DIR, 'bar_by_education.png'))
    plt.close()
    print('Saved images/bar_by_education.png')

def plot_line_by_country(df):
    # Ensure education level order is Primary -> Secondary -> Tertiary
    order = ['Primary', 'Secondary', 'Tertiary']
    df['Education_Level'] = pd.Categorical(df['Education_Level'], categories=order, ordered=True)
    pivot = df.pivot(index='Education_Level', columns='Country', values='Employment_Rate')
    pivot.plot(marker='o')
    plt.title('Education Level vs Employment Rate (by Country)')
    plt.xlabel('Education Level')
    plt.ylabel('Employment Rate (%)')
    plt.savefig(os.path.join(IMAGES_DIR, 'line_by_country.png'))
    plt.close()
    print('Saved images/line_by_country.png')

def build_and_evaluate_model(df):
    # We'll encode Education_Level as ordinal and Country as one-hot
    X = df[['Country', 'Education_Level']].copy()
    y = df['Employment_Rate'].values

    # Preprocessing: Country -> OneHot, Education_Level -> Ordinal
    ordinal = OrdinalEncoder(categories=[['Primary', 'Secondary', 'Tertiary']])
    onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(transformers=[
        ('edu_ord', ordinal, ['Education_Level']),
        ('country_ohe', onehot, ['Country'])
    ], remainder='drop')

    model = Pipeline(steps=[
        ('pre', preprocessor),
        ('lr', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('\n--- Model Evaluation ---')
    print(f'MSE: {{mse:.4f}}')
    print(f'R2:  {{r2:.4f}}')

    # Show a simple comparison dataframe
    comp = pd.DataFrame({{'Country': X_test['Country'].values, 'Education_Level': X_test['Education_Level'].values,
                          'Actual': y_test, 'Predicted': y_pred}})
    print('\n--- Predictions (test set) ---')
    print(comp)

    # Save predictions to CSV and a small plot
    comp.to_csv(os.path.join(PROJECT_ROOT, 'predictions.csv'), index=False)
    try:
        comp_plot = comp.copy()
        comp_plot['Index'] = range(len(comp_plot))
        comp_plot.plot(x='Index', y=['Actual', 'Predicted'], marker='o', linestyle='-')
        plt.title('Actual vs Predicted Employment Rate (test set)')
        plt.ylabel('Employment Rate (%)')
        plt.savefig(os.path.join(IMAGES_DIR, 'actual_vs_predicted.png'))
        plt.close()
        print('Saved images/actual_vs_predicted.png')
    except Exception as e:
        print('Could not save prediction plot:', e)

def main():
    df = load_data()
    basic_info(df)
    plot_bar_by_education(df)
    plot_line_by_country(df)
    build_and_evaluate_model(df)
    print('\nAll done. Check the images/ folder and predictions.csv')

if __name__ == '__main__':
    main()
