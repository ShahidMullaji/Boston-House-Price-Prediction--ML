# Boston-House-Price-Prediction--ML

A machine learning project that predicts the median value of homes in Boston neighbourhoods, with a Streamlit web app for live predictions.

I trained a Random Forest regression model on the classic Boston Housing dataset and wrapped it in a simple UI, so anyone can enter a neighbourhood's details and get a price estimate instantly.

## Dataset

- **Source:** Boston House Price Data (506 rows, 14 columns)
- **Target:** `PRICE` (median home value in $1000s)
- **Features (13):**

| Column | Description |
|---|---|
| CRIM | Per capita crime rate |
| ZN | Share of residential land zoned for large lots |
| INDUS | Share of non-retail business acres |
| CHAS | Borders the Charles River (1) or not (0) |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Share of owner-occupied units built before 1940 |
| DIS | Distance to Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Property tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio |
| B | Demographic index based on the Black population share |
| LSTAT | Share of lower-status population |

## What I did

1. Explored the data and checked for nulls and duplicates
2. Looked at the correlation heatmap. `RM` (about +0.7) and `LSTAT` (about -0.7) have the strongest relationship with price, while `RAD` and `TAX` are highly correlated with each other (about 0.9)
3. Split the data 80/20 into train and test sets
4. Trained a Random Forest regressor (no scaling needed, since it is tree-based)
5. Evaluated on the test set using R² and MAE
6. Saved the model with `joblib` and built the Streamlit app around it

## Results

Running `train.py` prints the test metrics:

| Metric | Score |
|---|---|
| R² | 0.89|
| MAE |1.96|

## Project structure

```
.
├── Boston House Price Data.csv   # dataset
├── train.py                      # trains the model, saves model.pkl and column.pkl
├── app.py                        # Streamlit web app
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/ShahidMullaji/boston-house-price-prediction.git
cd boston-house-price-prediction

pip install -r requirements.txt

python train.py          # creates model.pkl and column.pkl
streamlit run app.py     # opens the app in your browser
```

## App

Enter the 13 neighbourhood features in the form and click **Predict Price**. The app returns the estimated price both in dollars and in $1000s.


## Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib

## Notes

- The model is trained on a small, old dataset (1970s Boston), so treat predictions as a learning demo, not real-world valuations.
- The `B` feature in this dataset is based on a racial demographic formula and is considered ethically problematic. It is kept here only to match the original dataset, and I would drop it in any real-world use.

## Author

**Mohammad Shahid Mullaji**
[GitHub](https://github.com/ShahidMullaji) | [LinkedIn](https://linkedin.com/in/mohammad-shahid-7bba45290)
