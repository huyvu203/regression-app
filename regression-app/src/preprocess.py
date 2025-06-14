from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from data_loader import load_dataset

df = load_dataset("/home/huy/code/regression-app/data/california-housing.csv")

# Separate target and features
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Select numerical and categorical features
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = ["ocean_proximity"]

# Pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combining both
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])