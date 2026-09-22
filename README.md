# House Price Prediction & Analysis

A Streamlit app that explores a house-price dataset, trains and compares
several regression models, and predicts the price of a house you describe.

## Files
- `app.py` — the Streamlit app (Overview, Data Explorer, Analysis, Model
  Training, Predict Price pages)
- `model_utils.py` — data cleaning + model training/prediction (no Streamlit
  import, so it can be tested on its own)
- `charts.py` — all matplotlib charts
- `House_Price_Prediction_Dataset_Analysis.csv` — the dataset (2000 houses,
  columns: Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location,
  Condition, Garage, Price)
- `requirements.txt` — Python packages needed

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (all files, including the CSV, in the
   repo root — do not add the CSV to `.gitignore`).
2. Go to https://share.streamlit.io → **New app**.
3. Pick the repo, branch `main`, and main file path `app.py`.
4. Click **Deploy**. First build takes a minute or two.

## Note on model accuracy
In this dataset, Price has essentially no statistical relationship with
Area, Bedrooms, Bathrooms, Floors, or YearBuilt (correlations are all near
0, R² for every model is near 0 on test data). The app shows this clearly
on the Model Training page rather than hiding it — every prediction is
close to the dataset's average price, and that is the honest result for
this particular data, not an error in the code.
