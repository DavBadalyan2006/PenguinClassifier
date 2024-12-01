import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

st.title("🐧 Penguin Classification System")
st.write("""
This application classifies penguins into species based on their physical characteristics.
Adjust the input features below to see predictions from multiple models.
Currently developed for numerically categorized data, will soon develop for image-based.
""")


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    data = pd.read_csv(url)
    return data


data = load_data()
data = data.dropna()

data['sex'] = data['sex'].str.strip().str.capitalize()

le_species = LabelEncoder()
le_species.fit(data['species'])
data['species_encoded'] = le_species.transform(data['species'])

le_island = LabelEncoder()
le_island.fit(data['island'])
data['island_encoded'] = le_island.transform(data['island'])

le_sex = LabelEncoder()
le_sex.fit(data['sex'])
data['sex_encoded'] = le_sex.transform(data['sex'])

X = data[['island_encoded', 'bill_length_mm', 'bill_depth_mm',
          'flipper_length_mm', 'body_mass_g', 'sex_encoded']]
y = data['species_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# To see model performance uncomment the below code lines
# from sklearn.metrics import accuracy_score
# st.header("Model Performance")
# model_accuracy = {}
#
# for name, model in models.items():
#     y_pred = model.predict(X_test_scaled)
#     acc = accuracy_score(y_test, y_pred)
#     model_accuracy[name] = acc
#     st.write(f"**{name} Accuracy:** {acc:.2f}")
#
# fig_acc, ax_acc = plt.subplots()
# sns.barplot(x=list(model_accuracy.keys()), y=list(model_accuracy.values()), ax=ax_acc)
# plt.ylabel('Accuracy')
# plt.ylim(0, 1)
# plt.title('Model Comparison')
# st.pyplot(fig_acc)


st.header("Make a Prediction")

unique_islands = data['island'].unique()
unique_sexes = data['sex'].unique()

def user_input_features():
    island = st.selectbox("Island", unique_islands)
    sex = st.selectbox("Sex", unique_sexes)
    bill_length_mm = st.slider(
        "Bill Length (mm)",
        float(data['bill_length_mm'].min()),
        float(data['bill_length_mm'].max()),
        float(data['bill_length_mm'].mean())
    )
    bill_depth_mm = st.slider(
        "Bill Depth (mm)",
        float(data['bill_depth_mm'].min()),
        float(data['bill_depth_mm'].max()),
        float(data['bill_depth_mm'].mean())
    )
    flipper_length_mm = st.slider(
        "Flipper Length (mm)",
        float(data['flipper_length_mm'].min()),
        float(data['flipper_length_mm'].max()),
        float(data['flipper_length_mm'].mean())
    )
    body_mass_g = st.slider(
        "Body Mass (g)",
        float(data['body_mass_g'].min()),
        float(data['body_mass_g'].max()),
        float(data['body_mass_g'].mean())
    )

    try:
        island_encoded = le_island.transform([island])[0]
    except ValueError:
        st.error(f"Unexpected island value: {island}. Please select a valid option.")
        return None

    try:
        sex_encoded = le_sex.transform([sex])[0]
    except ValueError:
        st.error(f"Unexpected sex value: {sex}. Please select a valid option.")
        return None

    input_data = {
        'island_encoded': island_encoded,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex_encoded': sex_encoded
    }
    features = pd.DataFrame(input_data, index=[0])
    return features


input_df = user_input_features()

if input_df is not None:
    input_scaled = scaler.transform(input_df)

    st.subheader("Prediction Results")

    for name, model in models.items():
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        predicted_species = le_species.inverse_transform(prediction)[0]

        st.write(f"**{name} Prediction:** {predicted_species}")

        prob_df = pd.DataFrame(prediction_proba, columns=le_species.classes_)
        prob_df = prob_df.transpose().rename(columns={0: 'Probability'})
        st.write(f"**{name} Prediction Probability:**")
        st.write(prob_df)

        if hasattr(model, 'feature_importances_'):
            st.write(f"**{name} Feature Importances:**")
            importances = model.feature_importances_
            feature_names = X.columns
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            fig_imp, ax_imp = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_imp_df, ax=ax_imp)
            plt.title(f'{name} Feature Importances')
            st.pyplot(fig_imp)
        elif hasattr(model, 'coef_'):
            st.write(f"**{name} Feature Coefficients:**")
            coefficients = model.coef_[0]
            feature_names = X.columns
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            }).sort_values(by='Coefficient', ascending=False)

            fig_coef, ax_coef = plt.subplots()
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax_coef)
            plt.title(f'{name} Feature Coefficients')
            st.pyplot(fig_coef)
        else:
            st.write(f"**{name} does not provide feature importances or coefficients.**")


st.markdown("---")
st.write("© 2024 Penguin Classification System | Built with Streamlit and Scikit-learn")
