# Penguin Classification System

## Overview
The **Penguin Classification System** is a Streamlit-based web application designed to classify penguins into their respective species based on physical characteristics such as bill length, bill depth, flipper length, body mass, island of origin, and sex. The application demonstrates predictions using multiple machine learning models and provides insights into the model's performance and feature importance.

## Features
1. **Interactive Interface**: Users can select input features via sliders and dropdowns to simulate predictions.
2. **Multiple Models**: Logistic Regression, Random Forest Classifier, and Support Vector Classifier (SVC) are implemented.
3. **Prediction Probabilities**: Displays the prediction probabilities for each species.
4. **Model Insights**: Shows feature importances or coefficients for applicable models.
5. **Data Handling**: Encodes categorical variables and scales numerical data for model compatibility.
6. **Performance Metrics**: Includes accuracy comparison of models (code provided but commented out).

## Installation
### Prerequisites
- Python 3.8 or later
- `pip` package manager

### Required Libraries
The required libraries and their versions are:
```
streamlit==1.25.0
pandas==1.5.3
seaborn==0.12.2
matplotlib==3.7.2
scikit-learn>=1.3.0,<1.4.0
numpy>=1.23.0,<2.0.0
pillow>=9.0.0
```

### Steps
1. Clone the repository:
   ```bash
    git clone https://github.com/DavBadalyan2006/PenguinClassifier/
    cd PenguinClassifier
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Access the app in your browser at `http://localhost:8501`.

## Usage
1. Start the application by running the Streamlit command.
2. Adjust the input sliders and dropdowns to specify the penguin's features.
3. View predictions from Logistic Regression, Random Forest, and SVC models.
4. Explore prediction probabilities and feature importance/coefficient plots for the models.

## Data
The application uses the Palmer Penguins dataset, hosted by Seaborn:
- URL: [https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)
- Features used:
  - `bill_length_mm`
  - `bill_depth_mm`
  - `flipper_length_mm`
  - `body_mass_g`
  - `island`
  - `sex`
- Target variable: `species`

## Key Components
### Data Preprocessing
- Handles missing data by dropping rows with `NaN` values.
- Encodes categorical variables (`island`, `sex`, and `species`) using `LabelEncoder`.
- Scales numerical features using `StandardScaler`.

### Machine Learning Models
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Classifier (SVC)**

Each model is trained on a standardized dataset split into training (80%) and testing (20%) sets.

### User Interaction
- Input features are provided via Streamlit widgets (sliders and dropdowns).
- Predictions and model outputs are dynamically displayed based on user input.

## Customization
- To add or modify models, update the `models` dictionary in the code.
- To display additional performance metrics, uncomment the section labeled "Model Performance".

## Limitations
- The dataset is small and not suitable for real-world deployment without further enhancement.
- Models may not generalize well due to limited data.
- Feature importances are not available for models like SVC.

## Future Enhancements
1. Add support for image-based penguin classification.
2. Improve model generalization using cross-validation.
3. Implement additional machine learning models (e.g., Gradient Boosting, Neural Networks).
4. Incorporate unsupervised learning for exploratory data analysis.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Palmer Penguins dataset: [https://allisonhorst.github.io/palmerpenguins/](https://allisonhorst.github.io/palmerpenguins/)
- Developed using Streamlit and Scikit-learn.

## Contact
For questions or suggestions, please reach out to:
- **Developer**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [Your GitHub Link]
