import streamlit as st
import pandas as pd
import numpy as np
from mistral_call import filter_shopping_list

st.set_page_config(layout="centered")

st.title('ReMi - Recipe Mistral')



data_df = pd.DataFrame(
    {
        "To buy": ["potatoes", "eggs", "eggs", "яйца"],
        "Amount": [3, 12, 2, 1],
        "Unit": ["kg", "pieces", "counts", "packs"],
        "Done": [False, False, False, False]
    }
)

if 'data_df' not in st.session_state:
    st.session_state['data_df'] = data_df

input_list = st.text_area(
    "Insert your list here! 🐭",
    """✓ рыба
    ✓ листы для шаурмы или листья салата
    ✓ сухарики
    ✓ фрукты для салата и на десерт 
    ✓ авокадо для завтрака
    ✓ 350 g d' oignon
    ✓ авокадо 
    ✓ огурец
    ✓ 350 g de poivron de couleur rouge et vert
    ✓ 350 g de courgette
    ✓ 500 g de tomate bien mûres
    """
    )



if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button("Organize shopping list", on_click=click_button)

if st.session_state.clicked:
    data_df = filter_shopping_list(data_df)


data_df = st.data_editor(data_df,
    column_config={
        "favorite": st.column_config.CheckboxColumn(
            "Done",
            help="Select your items that you have already bought",
            default=False,
        )
    },
    disabled=["To buy", "Amount", "Unit"],
    hide_index=True,
    num_rows="dynamic",
    #on_change=update,
)

data_df