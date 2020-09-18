import streamlit as st

# utils
import joblib
import os
import hashlib
# passlib,bcrypt

# EDA Packages
import pandas as pd
import numpy as np

# Data Viz Package
import matplotlib
# Data Viz Package
import matplotlib
from networkx.drawing.tests.test_pylab import plt

matplotlib.use('Agg')

# image for background
from PIL import Image
import os
st.image(Image.open(os.path.join('arashimage.jpg')))


# creatre the grey color for data set
def highlight_survived(s):
    return ['background-color: grey']*len(s)


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Health Prediction App </h1>
		<h5 style="color:white;text-align:center;">Hospital </h5>
		</div>
		"""




descriptive_message_temp ="""
	<div style="background-color:rgba(218, 218, 218, 0.5);overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Definition</h3>
		<p>HCLS Analytics will help hospitals reduce the number of resources currently used in the hospitals by proactively interacting with the patient.</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:rgba(218, 218, 218, 0.5);overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


def main():
    #st.title('Health Predictions APP')
    st.markdown(html_temp.format('royalblue'), unsafe_allow_html=True)


    Menu=['Home','Login']
    subMenu= ['Plot']

    choice = st.sidebar.selectbox('Menu', Menu)
    if choice == 'Home':
        st.markdown(descriptive_message_temp, unsafe_allow_html=True)



    elif choice == 'Login':
        username = "Ismiletechnologies"
        password = st.sidebar.text_input('password', type='password')
        if st.sidebar.checkbox('Login'):

            if password == "12345":

                st.success("Welcome to {}".format(username))



                task = st.selectbox("Task", subMenu)
                if task == "Plot":
                    st.subheader("Data Visualization Plot")
                    df = pd.read_csv("clean heart.csv")
                    st.dataframe(df)


                    #show the target
                    if st.checkbox('Target 1 is more than 0 '):
                        df['target'].value_counts().plot(kind='bar')
                        st.pyplot()

                    # Freq Dist Plot
                    if st.checkbox('Between all ranges, range between 50 and 60 has the most amount of patients'):
                        freq_df = pd.read_csv("heart_frequency.csv")
                        st.bar_chart(freq_df['count'])


                    #correlation
                    import seaborn as sns
                    if st.checkbox('show correlation plot with Matplotlib'):
                        st.write(sns.heatmap(df.corr()))
                        st.pyplot()


                    if st.checkbox('Show bar chart plot'):
                        sns.countplot(data=df, y='target', palette='hls', hue='sex')
                        st.pyplot()


                    if st.checkbox('Area chart'):
                        all_columns = df.columns.to_list()
                        feat_choices = st.multiselect('choose a  features', all_columns)
                        new_df = df[feat_choices]
                        st.area_chart(new_df)
                        st.markdown(prescriptive_message_temp, unsafe_allow_html=True)









            else :
                st.warning("incorrect username/password")
























if __name__ == '__main__':
    main()











