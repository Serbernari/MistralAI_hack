import streamlit as st
import pandas as pd
import numpy as np
from mistral_call import mistral_compare_items, create_csv

st.set_page_config(layout="centered")

st.title('ReMi - Recipe Mistral')

form = st.form(key='my_form')
input_list = form.text_area(label="Insert your list here! 🐭")
submit_button = form.form_submit_button(label='Make shopping list')



if 'data_df' not in st.session_state and submit_button:
    data_df = create_csv(input_list)    
    st.session_state['data_df'] = data_df


if 'clicked_organize' not in st.session_state:
    st.session_state.clicked_organize = 0

def click_button():
    st.session_state.clicked_organize += 1

st.button("Organize shopping list", on_click=click_button)

generation = st.session_state.setdefault("generation", 0)

def on_rows_change():
    st.info(f"rows updated: generation={st.session_state.generation}")
    st.session_state.generation += 1


if st.session_state.clicked_organize == 1:
    st.session_state['data_df'] = mistral_compare_items(st.session_state['data_df'])
    st.session_state.clicked_organize += 1

prev_key = f"data.{generation-1}"

if prev_key in st.session_state:
    pass
try:
    st.session_state['data_df'] = st.data_editor(st.session_state['data_df'],
        column_config={
            "Done": st.column_config.CheckboxColumn( #favorite?
                "Done",
                help="Select your items that you have already bought",
                default=False,
            )
        },
        disabled=["Item", "Amount", "Unit"],
        hide_index=True,
        on_change=on_rows_change
    )
except KeyError:
    pass
