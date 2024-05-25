import streamlit as st
import pandas as pd
import numpy as np
from mistral_call import filter_shopping_list, create_csv

st.set_page_config(layout="centered")

st.title('ReMi - Recipe Mistral')

input_list = st.text_area(
    "Insert your list here! 🐭",
    )

data_df = create_csv(input_list)

if 'data_df' not in st.session_state:
    st.session_state['data_df'] = data_df

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
    disabled=["Item", "Amount", "Unit"],
    hide_index=True,
    num_rows="dynamic",
    #on_change=update,
)
