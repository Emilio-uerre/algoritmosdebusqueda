import streamlit as st

st.title('Sumadora de dos números')

numero1 = st.number_input('Ingresa el primer número', value=0)
numero2 = st.number_input('Ingresa el segundo número', value=0)

suma = numero1 + numero2

st.write(f'La suma de {numero1} + {numero2} es: {suma}')
