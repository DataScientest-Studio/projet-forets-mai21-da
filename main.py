import streamlit as st
import page_introduction
import page_modelisation
import page_exploration
import page_prediction



#def app():
   # st.title('Projet forêt')


#app = MultiApp()

# Add all your application here
#app.add_app("Projet forêt", newapp.app)








st.title('Emissions et Forets')

st.text('''Cette application montre le travail effectué par le groupe Forets.''')


# st.code('SELECT * FROM table', language='sql')

# input_select_box = st.selectbox(
#     label='choisir une valeur',
#     options=['Bonjour', 'Le', 'Monde'])

# st.text(input_select_box)

# import numpy as np

# nb_points = st.slider(
#     label='nombre de points',
#     min_value=2, max_value=100,
#     value=50,
#     step=1
#     )

# X = np.random.normal(size=nb_points)

# y = np.random.normal(size=nb_points)

# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10, 10))
# plt.scatter(X, y)

# st.pyplot(fig)

# st.sidebar.header('Menu')

menu_selected = st.sidebar.selectbox(
    label='Choix de menu',
    options=['Introduction',
             'Exploration',
             'Modélisation',
             'Prediction']
    )

if menu_selected == 'Introduction':
    page_introduction.render()
elif menu_selected == 'Modélisation':
    page_modelisation.render()
elif menu_selected == 'Exploration':
    page_exploration.render()
elif menu_selected == 'Prediction':
    page_prediction.render()