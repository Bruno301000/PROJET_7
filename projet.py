import streamlit as st
import pandas as pd
import shap
from shap.plots._beeswarm import summary_legacy
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache
def load_data():
	data= pd.read_csv('test_df.csv', index_col = [0])
	
	return data

df_test_rf=load_data()

# rf predictions
model_rf= joblib.load('RF_grid_model_v2.joblib')

list_cust = list(df_test_rf['SK_ID_CURR'])

st.sidebar.title("Customer for prediction ")
st.sidebar.subheader('Research a Customer ID')
select_box =st.sidebar.selectbox("Customer / Feature name  SK_ID_CURR",list_cust)
if st.sidebar.button('select customer'):
	st.write('Customer choosen',select_box)


st.title("CREDIT SCORING WEB APPLICATION ")

# Load data app_test_streamlit_v3
@st.cache(allow_output_mutation=True)
def load_data_brut():
	data_brut= pd.read_csv('app_test_streamlit_v3.csv',index_col = [0])
	return data_brut

df_brut =load_data_brut()

# Informations basiques sur le client
st.subheader('1. Customer basic informations')
cust_infos= pd.DataFrame(df_brut.loc[df_brut['SK_ID_CURR']==select_box,['CODE_GENDER','AGE','YEARS_EMPLOYED','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']])
st.dataframe(data=cust_infos)
st.write("")
st.write("")
st.write("")


#Prediction to be unpaid
st.subheader('2. Prediction  /  (Random Forest model)')
cust_id2= st.number_input('Optional manual enter / customer ID', min_value =100001, value =round(select_box))
left_column, right_column = st.columns(2)
with left_column:
	st.subheader('A- Probability to be unpaid   :')
with right_column:
	st.subheader('B- Predicted class')

df_single_cust=df_test_rf[df_test_rf['SK_ID_CURR']==cust_id2]
y_pred_rf = int(model_rf.predict(df_single_cust))
y_pred_proba= np.round((float(model_rf.predict_proba(df_single_cust)[:,1])*100),2)
left_column, right_column = st.columns(2)
with left_column:
	if y_pred_proba<=30:
		st.write(y_pred_proba, '%  :  low probability to be unpaid')
	elif (y_pred_proba>30) & (y_pred_proba<=49.99):
		st.write(y_pred_proba, '%  >  likely to be paid but risks exist')
	else:
		st.write(y_pred_proba, '%  > strong risks to be unpaid')


with right_column:
	if y_pred_rf == 0:
		st.success('class 0')
	else:
		st.warning('class 1')

st.write('For your information : ')
left_column, right_column = st.columns(2)
with left_column:
	st.info('class 0 = prediction to be paid ')
with right_column:
	st.info('class 1 = prediction to be unpaid')

st.write("")
st.write("")
# Explanations of the model
st.subheader('3. Details of involved features in the prediction of the customer selected')
st.write('probability  displays between 0 and 1 /  to get on % as above, please multiply by 100 ')
row_to_show = df_single_cust.index.item()
data_for_prediction = df_test_rf.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


# Calculate Shap values

explainer = shap.TreeExplainer(model_rf)
shap_values = explainer.shap_values(data_for_prediction)
#shap.initjs()

st.subheader('Force plot')

force_plot = shap.force_plot(np.round(explainer.expected_value[1],3),
                    np.round(shap_values[1],3),
                    np.round(data_for_prediction,3),
                    matplotlib=True,
                    show=False)

st.pyplot(force_plot)



#Scatterplot customer by class
st.subheader( "4. Overview  /  Customers by class with probability to be unpaid")
y_pred_global= model_rf.predict(df_test_rf)
y_pred_proba_global = model_rf.predict_proba(df_test_rf)[:,1]
df_test_rf_pred= df_test_rf.copy()
df_test_rf_pred['PREDICTED CLASS']=y_pred_global
df_test_rf_pred['PROBABILITY_%'] = np.round(y_pred_proba_global*100)

fig = px.scatter(df_test_rf_pred, x='SK_ID_CURR', y='PROBABILITY_%', color='PREDICTED CLASS',
	hover_data=['SK_ID_CURR'],marginal_y ='box')

st.plotly_chart(fig)



# Feature importances of the model
model_rf.predict_proba(data_for_prediction_array)

st.subheader('5. Features importances for the entire model')

feat_importances = pd.Series(model_rf.feature_importances_, index=df_test_rf.columns)
fig=px.bar(feat_importances.nlargest(15))
st.plotly_chart(fig)
st.write("")
st.write("")
st.write("")

# Global statistics from the dataset

# Creation colonne PROBABILITY_UNPAID_%
df_brut['PROBABILITY_UNPAID_%']=np.round(df_test_rf_pred['PROBABILITY_%'])

# Sample of df_brut for statistics
#df_sample=df_brut.sample()

for col in df_brut:
    if (df_brut[col].isna().sum()!=0)&(df_brut[col].dtype =='object'):
        df_brut=df_brut.drop(col, axis =1, inplace = True)

for col in df_brut:
    if (df_brut[col].dtype !='object'):
        df_brut[[col]]=df_brut[[col]].fillna(df_brut[[col]].median())


# Feature N°1 and feature N°2 to be choosen

st.sidebar.title('Global statistics')

st.subheader('6. Global statistics')
feature1 =st.sidebar.subheader("Feature n°1 to be selected")
select_box_feature1 =st.sidebar.selectbox("",df_brut.columns)

if st.sidebar.button('select Feature N°1'):
	st.write('Feature N°1 choosen',select_box_feature1)

feature2 =st.sidebar.subheader("Feature n°2 to be selected")
select_box_feature2 =st.sidebar.selectbox("",df_brut.columns.sort_values(ascending=False))
if st.sidebar.button('select Feature N°2'):
	st.write('Feature N°2 choosen',select_box_feature2)



# Graphique feature 1
left_column, right_column = st.columns(2)


with left_column:
	st.subheader("Histogram feature 1")
	fig = px.histogram(df_brut, x=select_box_feature1,marginal ='box', barmode ='group',
		                hover_data=['SK_ID_CURR','AGE', 'YEARS_EMPLOYED','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','PROBABILITY_UNPAID_%'])
	st.plotly_chart(fig)


# Graphique feature 2
with right_column:
	st.subheader("Histogram feature 2")
	fig = px.histogram(df_brut, x=select_box_feature2, color = 'CODE_GENDER',marginal ='box',barmode ='group',color_discrete_sequence =['orange','green']*3,
		               hover_data=['SK_ID_CURR','AGE', 'YEARS_EMPLOYED','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','PROBABILITY_UNPAID_%'])
	st.plotly_chart(fig)


# Test graphique 2 features 
st.subheader("Feature N°1  combines with feature N°2 / per Class : prediction to be paid or unpaid" )
df_brut['PREDICTED_CLASS']=np.round(df_test_rf_pred['PREDICTED CLASS'])
df_brut['PREDICTED_CLASS']=df_brut['PREDICTED_CLASS'].replace({0:'paid', 1:'unpaid'})
#df_brut['PROBABILITY_UNPAID_%']=np.round(df_test_rf_pred['PROBABILITY_%'])

fig = px.scatter(df_brut, x=select_box_feature1, y=select_box_feature2, color="PREDICTED_CLASS",hover_data=['SK_ID_CURR','AGE', 'YEARS_EMPLOYED',
	'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','PROBABILITY_UNPAID_%'],opacity=0.1,marginal_x='box', marginal_y ='box')
st.plotly_chart(fig)






