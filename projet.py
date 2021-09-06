import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import shap
import math
from shap.plots._beeswarm import summary_legacy
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

@st.cache
def load_data(nrows):
	data= pd.read_csv('test_df.csv',index_col = [0])
	return data



df_test_rf=load_data(4500)


# rf predictions
model_rf= joblib.load('RF_model.joblib')

#df_test_rf= pd.read_csv('test_df.csv',index_col = [0])

list_cust = list(df_test_rf['SK_ID_CURR'])

st.sidebar.title("Customer")
st.sidebar.subheader('Research a Customer ID')
select_box =st.sidebar.selectbox("customer",list_cust)
if st.sidebar.button('select customer'):
	st.write('customer choosen',select_box)


st.title("CREDIT SCORING ")

# Informations basiques sur le client
@st.cache
def load_data_brut(nrows):
	data= pd.read_csv('app_test_streamlit.csv')
	return data

df_brut = load_data_brut(4500)
#df_brut= pd.read_csv('app_test_streamlit.csv')

st.subheader('Customer basic information')

cust_infos= pd.DataFrame(df_brut.loc[df_brut['SK_ID_CURR']==select_box,['CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']])
cust_infos['AGE']= (df_brut['DAYS_BIRTH']/-365).astype(int)
cust_infos['AMT_INCOME_TOTAL']= cust_infos['AMT_INCOME_TOTAL'].astype(int)
cust_infos['AMT_CREDIT']= cust_infos['AMT_CREDIT'].astype(int)
cust_infos['AMT_ANNUITY']= cust_infos['AMT_ANNUITY'].astype(int)
cust_infos['YEARS_EMPLOYED']= (df_brut['DAYS_EMPLOYED']/-365).astype(int)

st.dataframe(data=cust_infos)

#st.subheader('the customer ID  ')
cust_id2= st.number_input('(random forest model)/ customer ID', min_value =100001, value =round(select_box))
st.subheader('Probability to be unpaid   :')

df_single_cust=df_test_rf[df_test_rf['SK_ID_CURR']==cust_id2]
y_pred_rf = int(model_rf.predict(df_single_cust))
y_pred_proba= np.round((float(model_rf.predict_proba(df_single_cust)[:,1])*100),2)
st.write(y_pred_proba, '%')

#st.write('class :', y_pred_rf)

if y_pred_rf == 0:
	st.success('class 0')
else:
	st.warning('class 1')
st.write('For your information : ')
st.info('class 0 = prediction to be paid ')
st.info('class 1 = prediction to be unpaid')


# Explanations of the model
st.subheader('Details of involved features in the prediction of the customer selected')
st.write('due to rounded figures -> prediction shown is likely to differ a litlle bit from above')
st.write('probability  display between 0 and 1 /  to get on % as above, please multiply by 100 ')
row_to_show = df_single_cust.index.item()
data_for_prediction = df_test_rf.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
# Create object that can calculate shap values
explainer = shap.TreeExplainer(model_rf)
# Calculate Shap values
shap_values_one = explainer.shap_values(data_for_prediction)
shap.initjs()
st.pyplot(shap.force_plot(np.round(explainer.expected_value[1],3), np.round(shap_values_one[1],3), np.round(data_for_prediction,3),matplotlib=True, figsize=(40,6)))
#st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
plt.clf()

# Feature importances of the model
model_rf.predict_proba(data_for_prediction_array)

st.subheader('Features importances for the entire model')
feat_importances = pd.Series(model_rf.feature_importances_, index=df_test_rf.columns)
feat_importances.nlargest(15).plot(kind='barh',figsize=(6,7)).invert_yaxis()
st.pyplot(feat_importances.nlargest(15).plot(kind='barh').invert_yaxis())

# Feature importances for each class with shap
st.subheader('Features importances detailed for each class')
explainer= shap.TreeExplainer(model_rf)
shap_value1=explainer.shap_values(df_test_rf)
shap_obj = explainer(df_test_rf.values.reshape(-1))

#st.pyplot(summary_legacy(shap_value1,df_test_rf))
st.pyplot(shap.summary_plot(shap_value1, df_test_rf))
#st.pyplot(shap.plots.beeswarm(shap_obj))

# Global statistics from the dataset
# slider avec une variable

st.sidebar.title('Global statistics')
feat_importances_best15= feat_importances.sort_values(ascending = False)[:15]

feature1 =st.sidebar.subheader("Select Feature n°1 from feature importancces")
select_box_feature1 =st.sidebar.selectbox("Feature N°1",feat_importances_best15.index)
if st.sidebar.button('select Feature N°1'):
	st.write('customer choosen',select_box_feature1)

feature2 =st.sidebar.subheader("Select Feature n°2 from feature importancces")
select_box_feature2 =st.sidebar.selectbox("Feature N°2",feat_importances_best15.index)
if st.sidebar.button('select Feature N°2'):
	st.write('customer choosen',select_box_feature2)

add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)


st.vega_lite_chart(df_brut, {
    'mark': {'type': 'circle', 'tooltip': True},
     'encoding': {
         'x': {'field': 'select_box_feature1', 'type': 'quantitative'},
 'y': {'field': 'select_box_feature2', 'type': 'quantitative'},
         'size': {'field': 'DAYS_BIRTH', 'type': 'quantitative'},
         'color': {'field': 'DAYS_BIRTH', 'type': 'quantitative'},
     },
 })


c = alt.Chart(df_brut).mark_circle().encode(
     x='select_box_feature1', y='select_box_feature2', size='DAYS_BIRTH',
      color='DAYS_BIRTH', tooltip=['select_box_feature1', 'select_box_feature2', 'DAYS_BIRTH'])

st.altair_chart(c, use_container_width=True)


#fig= plt.hist(df)	

#st.pyplot(fig)
#st.pyplot(title('select_box_feature1'), 
#st.pyplot(xlabel('select_box_feature1'))
#st.pyplot(ylabel('Count'))
#[i for i in df_test_rf.columns if i==select_box_feature2]
#plt.hist(df_test_rf[i], edgecolor = 'k')
#plt.title(select_box_feature2); plt.xlabel('Age (years)'); plt.ylabel('Count');


#streamlit.expander(label, expanded=False)


