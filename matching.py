from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from causallib.estimation import IPW
from causallib.preprocessing.transformers import MatchingTransformer
import pandas as pd

def perform_matching(df_encoded, confounds, prognostics, treatment, outcome):
    covariates = confounds + prognostics
    covariates = [col for col in covariates if col in df_encoded.columns]
    
    X = df_encoded[covariates]
    a = df_encoded[treatment] >= df_encoded[treatment].median()
    a = a.astype(int)
    y = df_encoded[outcome]

    # Scale the covariates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to a DataFrame
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    # Debug prints
    print("Covariates: ", covariates)
    print("X shape: ", X.shape)
    print("X types: ", X.dtypes)
    print("a shape: ", a.shape)
    print("a types: ", a.dtypes)
    print("y shape: ", y.shape)
    print("y types: ", y.dtypes)

    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    ipw = IPW(lr)
    ipw.fit(X_scaled, a)

    propMatrix = ipw.compute_propensity_matrix(X_scaled, a).to_dict(orient="records")

    def formatData(X, a, y, propMatrix):
        unadjustedData = []
        confounds = X.to_dict(orient="records")
        a.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        for i in range(len(confounds)):
            newDataInstance = confounds[i]
            newDataInstance['treatment'] = a[i]
            newDataInstance['propensity'] = propMatrix[i]
            newDataInstance['outcome'] = y[i]  # Add outcome to each instance
            unadjustedData.append(newDataInstance)
        return unadjustedData

    unadjustedCohort = formatData(X_scaled, a, y, propMatrix)
    matcher = MatchingTransformer(with_replacement=True, n_neighbors=1, caliper=2)
    matcher.fit(X_scaled, a, y)
    Xm, am, ym = matcher.transform(X_scaled, a, y)
    propMatrixm = ipw.compute_propensity_matrix(Xm, am).to_dict(orient="records")
    adjustedCohort = formatData(Xm, am, ym, propMatrixm)

    return adjustedCohort, unadjustedCohort