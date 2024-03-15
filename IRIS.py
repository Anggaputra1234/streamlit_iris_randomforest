import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing data
predictions = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions, target_names=iris.target_names)

# Create a Streamlit app
st.title('Iris Flower Classification with Random Forest (Muh. Angga Adi Putra)')
st.write('Model training and evaluation results:')
st.write(f'Training set size: {len(X_train)}')
st.write(f'Testing set size: {len(X_test)}')
st.write(f'Model accuracy: {accuracy}')
st.write('Confusion Matrix:')
st.write(conf_matrix)
st.write('Classification Report:')
st.write(class_report)

# Add user input for feature values
sepal_length = st.slider('Sepal length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider('Sepal width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider('Petal length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider('Petal width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Make a prediction
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display the prediction
species = iris.target_names[prediction][0]
st.write(f'The predicted species is {species}')
