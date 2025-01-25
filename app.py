import random
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from shapash.explainer.smart_explainer import SmartExplainer
import subprocess
import os
import webbrowser
import mlflow
import shap
import plotly.io as pio
from codecarbon import EmissionsTracker


st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Introduction','Visualization','Prediction', 'ML Flow & Feature Importance', 'Conclusion'])


fraud_data = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data_2.csv')

# DATA TRIMMING
def data_trimming():
  ## change categorical text into categorical numeric
  fraud_data['Payment Method'] = fraud_data['Payment Method'].astype('category').cat.codes
  fraud_data['Product Category'] = fraud_data['Product Category'].astype('category').cat.codes
  fraud_data['Device Used'] = fraud_data['Device Used'].astype('category').cat.codes

# Training data
def process_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Training data
def model_prediction(model, X_train, y_train, test_size=0.3, random_state=42):
  tracker = EmissionsTracker()
  # Logistic Regression
  if model == 'Logistic Regression':
    st.title("Logistic Regression")
    tracker.start()
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predict = logmodel.predict(X_test)
    emissions = tracker.stop()

  # KNN Algorithm
  elif model == 'K-Nearest Neighbor Algorithm':
    st.title("K-Nearest Neighbor")

    k_list = list(range(10, 105, 10))  # Range of k values to try
    grid = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': k_list}, cv=3, scoring='accuracy')
    tracker.start()
    grid.fit(X_train, y_train)
    emissions = tracker.stop()

    best_k = grid.best_params_['n_neighbors']
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    predict = best_knn.predict(X_test)

    # Displaying graph
    mean_test_scores = grid.cv_results_['mean_test_score']
    plt.figure(figsize=(10, 5))
    plt.plot(k_list, mean_test_scores, color='navy', linestyle='dashed', marker='o')
    plt.xlabel('K Number of Neighbors', fontdict={'fontsize': 15})
    plt.ylabel('Accuracy', fontdict={'fontsize': 15})
    plt.title('K NUMBER X ACCURACY', fontdict={'fontsize': 30})
    st.pyplot(plt)

    st.markdown("#### Evaluation for the best Kth value")

  # Decision Tree Classifier
  else:
    st.title("Decision Tree Classifier")
    tracker.start()
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    emissions = tracker.stop()
    max_depth = clf.tree_.max_depth
    st.write("Max Depth of Decision Tree:", max_depth)

  st.text("Estimated emissions for training the model: " + str(emissions) + " kg of CO2")

  return predict

def evaluate(y_test, predict):
  st.text("Accuracy: " + str(accuracy_score(y_test, predict)))
  st.text("Precision: " + str(precision_score(y_test, predict)))
  st.text("Recall: " + str(recall_score(y_test, predict)))
  st.text("F1 Score: " + str(f1_score(y_test, predict)))


if app_mode == "Introduction":

  st.image("eden_data.jpeg", use_column_width=True)
  st.title("Company Overview")
  st.markdown("Our client is Eden Data, one of the reknowned Cyber Security company in the USA that focuses on make internet a safe space to make online transactions. Recently, they observed that there has been a peak in the number of fraudulent activity on internet and to ensure the safety of their customers they want us to help analyse the features of potential fraudulent activity")

  st.markdown("##### Objectives")
  st.markdown("- Analyse the features for online transactions that contribute to fraudulent activity.")
  st.markdown("- Classify online transactions into potentially fraudulent or not")

  st.markdown("##### Dataset Description")
  st.write("This dataset mainly focuses on collecting key features of transaction data with a focus on fraud detection. The number of transactions recorded for this dataset is 23,634 transactions, and some of the key features of this dataset include Transaction Amount (integer return type), Transaction Date (integer return type), Is Fraudulent (binary return type), Billing Address (string return type), and more.")

  num = st.number_input('No. of Rows', 5, 10)

  head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
  if head == 'Head':
    st.dataframe(fraud_data.head(num))
  else:
    st.dataframe(fraud_data.tail(num))

  st.text('(Rows,Columns)')
  st.write(fraud_data.shape)

  st.markdown("##### Key Variables")
  st.markdown("- Transaction Amount")
  st.markdown("- Payment Method")
  st.markdown("- Product Category")
  st.markdown("- Customer Age")
  st.markdown("- Customer Location")
  st.markdown("- Device Used")

  st.markdown("From all these variables we aim to predict a price that the customers would be willing to pay for Vehicle Insurance.")
  st.markdown("Analysing the relationships between such as 'Vehicle Damage' and 'Previously_insured' with 'Response' will help us define our target audience.")
  st.markdown("Similarly, analysing demographics such as 'Region', 'Gender', 'Age' and their relationships with 'Price' will help us define a price point.")

  st.markdown("### Description of Data")
  st.dataframe(fraud_data.describe())
  st.markdown("Descriptions for all quantitative data **(rank and streams)** by:")

  st.markdown("Count")
  st.markdown("Mean")
  st.markdown("Standard Deviation")
  st.markdown("Minimum")
  st.markdown("Quartiles")
  st.markdown("Maximum")

  st.markdown("### Missing Values")
  st.markdown("Null or NaN values.")

  dfnull = fraud_data.isnull().sum()/len(fraud_data)*100
  totalmiss = dfnull.sum().round(2)
  st.write("Percentage of total missing values:",totalmiss)
  st.write(dfnull)
  if totalmiss <= 30:
    st.success("We have less then 30 percent of missing values, which is good. This provides us with more accurate data as the null values will not significantly affect the outcomes of our conclusions. And no bias will steer towards misleading results. ")
  else:
    st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

  st.markdown("### Completeness")
  st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

  st.write("Total data length:", len(fraud_data))
  nonmissing = (fraud_data.notnull().sum().round(2))
  completeness= round(sum(nonmissing)/len(fraud_data),2)

  st.write("Completeness ratio:",completeness)
  st.write(nonmissing)
  if completeness >= 0.80:
    st.success("We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")
  else:
    st.success("Poor data quality due to low completeness ratio( less than 0.85).")

elif app_mode == "Visualization":
  st.title("Looker Studio Dashboard")
  st.image("looker.png", use_column_width = True)
  link = "https://lookerstudio.google.com/u/0/reporting/c19e3ce3-5340-4050-9de4-4a73f0b777d6/page/zGPwD"
  st.markdown(f"[Open Looker Studio]({link})")



elif app_mode == "Prediction":

  data_trimming()

  selected_variables = st.multiselect('Choose variables', ['Transaction Amount', 'Payment Method', 'Product Category', 'Customer Age', 'Device Used'], default=['Transaction Amount'])
  model = st.selectbox('Choose the model', ['Logistic Regression', 'K-Nearest Neighbor Algorithm', 'Decision Tree Classifier'], index=0)

  X = fraud_data[selected_variables]
  y = fraud_data['Is Fraudulent']

  X_train, X_test, y_train, y_test = process_data(X, y)
  prediction = model_prediction(model, X_train, y_train)
  evaluate(y_test, prediction)

session_state = st.session_state
if "data_loaded" not in session_state:
    session_state.data_loaded = {"ml_flow": False, "feature_importance": False}

elif app_mode == "ML Flow & Feature Importance":

    data_trimming()
    X = fraud_data[['Transaction Amount', 'Payment Method', 'Product Category', 'Customer Age', 'Device Used']]
    y = fraud_data['Is Fraudulent']
    X_train, X_test, y_train, y_test = process_data(X, y)

    tab1, tab2 = st.tabs(["ML Flow", "Feature Importance"])

    tab1.subheader("ML Flow")

    def train_model(exp_name, model, X_train, X_test, y_train, y_test):

        try:
            if not session_state.data_loaded["ml_flow"]:
                experiment = mlflow.set_experiment(exp_name)
                with st.spinner("Training the model..."):
                    with mlflow.start_run(experiment_id=experiment.experiment_id):
    
                        # Finding the best model
                        if model == "dt":
                          model_instance = DecisionTreeClassifier()
                          param_grid = {'max_depth': [5, 10, 20, 30, 40], 'min_samples_leaf': [1, 2, 4]}
                        elif model == "lg":
                          model_instance = LogisticRegression()
                          param_grid = {'C': [0.01, 0.1, 1, 10, 50]}
    
                        # modelling the best model in knn
                        if model =="knn":
                          knn = KNeighborsClassifier(n_neighbors=60)
                          knn = knn.fit(X_train, y_train)
                          y_pred = knn.predict(X_test)
                        else:
                          grid_search = GridSearchCV(estimator=model_instance, param_grid=param_grid, cv=3)
                          grid_search.fit(X_train, y_train)
                          best_model = grid_search.best_estimator_
    
                          mlflow.log_params(grid_search.best_params_)
                          mlflow.sklearn.log_model(best_model, model)
                          mlflow.sklearn.save_model(best_model, (model+"_model"))
    
                          y_pred = best_model.predict(X_test)
    
                          # Calculate evaluation metrics
                          accuracy = accuracy_score(y_test, y_pred)
                          precision = precision_score(y_test, y_pred, average='macro')
                          recall = recall_score(y_test, y_pred, average='macro')
                          f1 = f1_score(y_test, y_pred, average='macro')
    
                          # Log metrics to MLflow
                          mlflow.log_metric("accuracy", accuracy)
                          mlflow.log_metric("precision", precision)
                          mlflow.log_metric("recall", recall)
                          mlflow.log_metric("f1", f1)
                            
                        session_state.data_loaded["ml_flow"] = True

        except mlflow.exceptions.MlflowException:
            return

    # train_model("Best_Model", "dt", X_train, X_test, y_train, y_test)
    # train_model("Best_Model", "lg", X_train, X_test, y_train, y_test)
    # train_model("Best_Model", "knn", X_train, X_test, y_train, y_test)
    tab1.image("ml_data.png", use_column_width=True)

    tab2.subheader("Feature Importance")

    # def generate_plots(X_train, y_train, X_test):
    #   with st.spinner("Generating Report..."):
    #     # feature importance for best model: log 0.01
    #     logmodel = LogisticRegression(C=0.01)
    #     logmodel = logmodel.fit(X_train, y_train)
    #     y_pred = logmodel.predict(X_test)

    #     # Reshaping y_pred and X_test according to Smart AI
    #     y_pred = pd.Series(y_pred)
    #     X_test = X_test.reset_index(drop=True)
    #     xpl = SmartExplainer(logmodel)
    #     xpl.compile(x=X_test, y_pred=y_pred)

    #     # Plot feature importance
    #     feature_importance_plot = xpl.plot.features_importance()
    #     pio.write_image(feature_importance_plot, "feature_importance_plot.png")

    #     # Randomly select a subset of indices and plot feature importance
    #     subset = random.choices(X_test.index, k=50)
    #     subset_plot = xpl.plot.features_importance(selection=subset)
    #     pio.write_image(subset_plot, "subset_plot.png")

    #     # Plot contribution plot
    #     contribution_plot = xpl.plot.contribution_plot('Transaction Amount')
    #     pio.write_image(contribution_plot, "contribution_plot.png")

    #     return feature_importance_plot, subset_plot, contribution_plot

    if tab2.button("Get Feature Importance"):
        # if not session_state.data_loaded["feature_importance"]:
        #     sample_data = fraud_data.sample(n=10000, random_state=42)
        #     X = sample_data[['Transaction Amount', 'Payment Method', 'Product Category', 'Customer Age', 'Device Used']]
        #     y = sample_data['Is Fraudulent']
        #     X_train, X_test, y_train, y_test = process_data(X, y)
        #     feature_importance_plot, subset_plot, contribution_plot = generate_plots(X_train, y_train, X_test)
        tab2.image("feature_importance.png")
        tab2.image("subset_plot.png")
        tab2.image("contribution_plot.png")
        



elif app_mode == "Conclusion":
  st.title("Simulator")

  data_trimming()

  X = fraud_data[['Transaction Amount', 'Payment Method', 'Product Category', 'Customer Age', 'Device Used']]
  y = fraud_data['Is Fraudulent']

  X_train, X_test, y_train, y_test = process_data(X, y)

  process_data(X, y)

  logmodel = LogisticRegression()
  logmodel.fit(X_train, y_train)

  # Convert user input to categorical codes
  payment_methods = {'PayPal': 0, 'Bank Transfer': 1, 'Credit Card': 2, 'Debit Card': 3}
  product_categories = {'clothing': 0, 'electronics': 1, 'health & beauty': 2, 'home & garden': 3, 'toys & games': 4}
  devices = {'Desktop': 0, 'Mobile': 1, 'Tablet': 2}

  user_input = []

  user_input.append(st.number_input('Enter the Transaction Amount', value=0))
  method = st.selectbox('Pick Payment Method', list(payment_methods.keys()), index=0)
  user_input.append(payment_methods[method])
  category = st.selectbox('Pick Product Category', list(product_categories.keys()), index=0)
  user_input.append(product_categories[category])
  user_input.append(st.number_input('Enter Customer Age', value=1, min_value=1, max_value=90))
  device = st.selectbox('Pick Device Used', list(devices.keys()), index=0)
  user_input.append(devices[device])

  # Make prediction using the trained model
  predict = logmodel.predict([user_input])
  if predict[0] == 0:
    st.success("Well done! You paid securely")
  else:
    st.error("You have been scammed!")
  st.title("To Conclude...")
  st.write("To conclude this final project our study mainly focused on the feature importance of fraud in e-commerce, and the study of several visual models through the use of looker to visualize general features of the dataset. To aid with this predictions, we focused on allowing the users to select parameters in order to study several factors like the CO2 emissions, accuracy, precision score, and more. Additionally we worked on training the models with ML flow, and launching it to get the best models. Coming to the end of this presentation, we want to address a few areas of improvement, in order to further improve the accuracy of this project. First, we would like to always maximize the completeness of our data set so, the ratio of non-missing values to total records, as it further improves the accuracy of all our models, so picking a suitable dataset is ideal. Additionally, typically, an F1 score > 0.9 is considered excellent, and below would be good or average, but our F1 score is greater than 1 which is probably an issue in the data set, which roots back to our main point of improvement which is choosing the most precise dataset. ")
  st.title("Areas Of Improvement...")
  st.write("Coming to the end of this presentation, we want to address a few areas of improvement, in order to further improve the accuracy of this project. First, we would like to always maximize the completeness of our data set so, the ratio of non-missing values to total records, as it further improves the accuracy of all our models, so picking a suitable dataset is ideal. Additionally, typically, an F1 score > 0.9 is considered excellent, and below would be good or average, but our F1 score is greater than 1 which is probably an issue in the data set, which roots back to our main point of improvement which is choosing the most precise dataset. ")
  st.title("Please Scan and Give Your Feedback")
  st.image("QR.png", use_column_width=True)