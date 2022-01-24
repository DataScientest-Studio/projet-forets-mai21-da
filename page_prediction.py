import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def predict(
        aviation_input,
        surface_foret_input,
        aviation_input2,
        surface_foret_input2):

    return np.random.uniform()


def render():

    a = st.slider(label='emission aviation',
                         min_value=100, max_value=10000)
    b = st.slider(label='surface forets',
                              min_value=100, max_value=10000)
    c = st.slider(label='emission aviation2',
                          min_value=100, max_value=10000)
    d = st.slider(
        label='surface forets2', min_value=100, max_value=10000)

    st.text(predict(a, b, c, d))

    
    growth_ratio_input = st.number_input(label='Growth ratio', min_value=.01, max_value=1., value=.1)
    
    a0 = 1
    b0 = 2
    c0 = 4
    d0 = 5
    
    liste_a = [a0 * (1 + growth_ratio_input) **i for i in range(10)]
    liste_b = [b0 * (1 + growth_ratio_input) **i for i in range(10)]
    liste_c = [c0 * (1 + growth_ratio_input) **i for i in range(10)]
    liste_d = [d0 * (1 + growth_ratio_input) **i for i in range(10)]
    
    
    fig = plt.figure(figsize=(20, 10))
    plt.plot(liste_a)
    plt.plot(liste_b)
    plt.plot(liste_c)
    plt.plot(liste_d)
    
    plt.legend()
    
    st.pyplot(fig)
    
    
    fig2 = plt.figure(figsize=(20, 10))
    plt.plot([predict(i, j, k, l) for i, j, k, l in zip(liste_a, liste_b, liste_c, liste_d)])
    
    st.pyplot(fig2  )
    

  