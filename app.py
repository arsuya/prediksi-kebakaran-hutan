import streamlit as st
import eda, prediction, home

st.set_page_config(page_title='Prediksi kebakaran hutan',
                   layout='centered',
                   initial_sidebar_state='expanded')

with st.sidebar:
    st.write('# Navigation')
    navigation = st.radio('Page', ['Home', 'EDA', 'Prediction'])
    st.markdown("---")

    st.markdown("Project Data Science oleh<br><a href='https://www.linkedin.com/in/arvinwibowo/'>Arvin Surya Wibowo</a>", unsafe_allow_html=True)

if navigation == 'Home':
    home.run()
elif navigation == 'EDA':
    eda.run()
else:
    prediction.run()