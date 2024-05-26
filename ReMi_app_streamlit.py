import streamlit as st
import pandas as pd
from mistral_call import mistral_compare_items, create_csv

# Set page configuration
st.set_page_config(page_title="ReMi", page_icon="🐭", layout="centered")

# Custom CSS to style the background of the table
st.markdown("""
    <style>
    .center-table {
        display: flex;
        justify-content: center;
    }
    .table-background {
        background-color: #f0f0f0; /* Change this color as needed */
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Display logo
st.image('logo_or_on_wh.png', width=150)

placeholder_text = """✓ Avocate
 ✓ 350 g d' oignon
 ✓ cucumber
 ✓ 350 g de poivron rouge
 ✓ 350 g de courgette
 ✓ 500 g de tomate
 ✓ arugula
 ✓ iceberg lettuce
 ✓ small tomatoes
 ✓ cooked chickpeas in a can
 ✓ tomato paste 500 ml
 ✓ canned corn 
 ✓ canned red beans 
 ✓ Onion 150 g 
 ✓ Canned tuna
 ✓ Chedar cheese
 ✓ сыр пармезан тертый 
 ✓ Chicken 1 kg 
 ✓ Tuna can 1
"""


# Form for input list
form = st.form(key='my_form')
input_list = form.text_area(label="Insert your list here!", placeholder=placeholder_text)
submit_button = form.form_submit_button(label='Make shopping list')

# Initialize session state for data DataFrame and button click count
if 'data_df' not in st.session_state and submit_button:
    try:
        if len(input_list) == 0:
            input_list = placeholder_text
        data_df = create_csv(input_list)
        st.session_state['data_df'] = data_df
    except Exception as e:
        st.error(f"Error creating shopping list: {e}")

if 'organize_click_count' not in st.session_state:
    st.session_state.organize_click_count = 0

# Function to handle organize button click
def click_button():
    st.session_state.organize_click_count += 1

st.button("Organize shopping list", on_click=click_button)

# Function to handle row changes in the data editor
def on_rows_change():
    st.session_state.generation += 1

# Organize shopping list if button clicked
if st.session_state.organize_click_count == 1:
    try:
        st.session_state['data_df'] = mistral_compare_items(st.session_state['data_df'])
        st.session_state.organize_click_count += 1
    except Exception as e:
        st.error(f"Error organizing shopping list: {e}")

# Display the data editor for the shopping list
generation = st.session_state.setdefault("generation", 0)
prev_key = f"data.{generation-1}"

col1, col2, col3 = st.columns([1, 20, 1])
# Wrap the data editor in a centered div with a background
st.markdown('<div class="center-table"><div class="table-background">', unsafe_allow_html=True)
with col2:
    if prev_key in st.session_state:
        pass
    try:
        changed_df = st.data_editor(st.session_state['data_df'],
            column_config={
                "Done": st.column_config.CheckboxColumn(
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
st.markdown('</div></div>', unsafe_allow_html=True)

st.divider()

# Display appropriate image based on user actions
with col3:
    if not submit_button and st.session_state.organize_click_count == 0:
        st.image('remi_question.png', width=350)
    elif submit_button and st.session_state.organize_click_count == 0:
        st.image('remi_happy.png', width=355)
    elif submit_button or st.session_state.organize_click_count >= 1:
        st.image('remi_super.png', width=340)
        if not st.session_state.get('balloons_shown', False):
            st.balloons()
            st.session_state.balloons_shown = True

st.markdown("<div style='text-align: center; margin-top: 50px;'>"
            "<p>Remi</p>"
            "<p>Enjoy your shopping list!</p>"
            "</div>", unsafe_allow_html=True)