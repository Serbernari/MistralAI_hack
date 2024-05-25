from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import streamlit as st

api_key = "dnDSPYh1mNwZyGRSQgjCPKQc144MfMng"
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

def get_mistral_compare_items(item_A, item_B):
    """
    Asks Mistral to compare two items from the list and say if it's the same thing 
    """
    #step 1
    chat_response = client.chat(
    model=model,
    messages=[ChatMessage(role="system", content='You will be asked questions about items to determine if they are the same name for a product inside grocery store. Specifically, you will need to assess if item_A and item_B are the same in terms of being found in the same section or location within the store. For example, "картошка" and "картофелина" should be considered the same since they can be bought in the same place, and you should answer "yes". Please answer with "yes" or "no"'),
              ChatMessage(role="user", content=f'If "{item_A}" and "{item_B}" are the same?')]
    )
    res1 = chat_response.choices[0].message.content
    
    # step 2
    chat_response2 = client.chat(
    model=model,
    messages=[ChatMessage(role="system", content='You will be given an answer to a certain question and you are need to summarize the answer as "yes" or "no". Answer only with "yes" or "no".'),
              ChatMessage(role="user", content=f'Does this text means "yes" or "no"? text:{res1}')]
    )
    res2 = chat_response2.choices[0].message.content
 
    if "yes" in res2.lower(): #need to properly regex this
        return True
    elif "no" in res2.lower():
        return False
    else:
        raise Exception("Mistral was unable to answer 'yes' or 'no'") 

def get_mistral_convert_units(item):
    """
    Asks Mistral to compare two items from the list and say if it's the same thing 
    """
    #step 1
    chat_response = client.chat(
    model=model, #
    messages=[ChatMessage(role="user", content=f'Please convert this unit into the International System of Units: "1 bottle of vodka" I need it to go to shop in France and I want all items to be represented in the same measurements. I want pack of milk represented as 1 liter, 1 potato as 0.15 kg and so on. Please provide short answer')])
    res1 = chat_response.choices[0].message.content
    
        # step 2
    chat_response2 = client.chat(
    model=model,
    messages=[ChatMessage(role="system", content='Extract number and SI unit from the given text. Format your text like "10 kilograms" or "1 litre"'),
              ChatMessage(role="user", content=f'text:{res1}')]
    )
    res2 = chat_response2.choices[0].message.content
    
            # step 3
    chat_response3 = client.chat(
    model=model,
    messages=[ChatMessage(role="system", content='Extract number and SI unit from the given text. Format your text like "10 kilograms" or "1 litre". Answer only with number and unit, nothing else.'),
              ChatMessage(role="user", content=f'text:{res2}')]
    )
    res3 = chat_response3.choices[0].message.content
    
    return res3

@st.cache_data
def filter_shopping_list(data_df):
    for i in data_df.index:
        for j in data_df.index:
            if i != j and data_df.loc[i, 'To buy'] is not None and data_df.loc[j, 'To buy'] is not None:
                if data_df.loc[i, 'To buy'].lower() == data_df.loc[j, 'To buy'].lower():
                    data_df.loc[i,'Amount'] += data_df.loc[j,'Amount']
                    data_df.loc[j, 'To buy'] = None
                else:
                    try:
                        if get_mistral_compare_items(data_df.loc[i, 'To buy'], data_df.loc[j, 'To buy']) == True:
                            data_df.loc[i,'Amount'] += data_df.loc[j,'Amount']
                            data_df.loc[j, 'To buy'] = None
                    except Exception as e:
                        print(e)
    data_df = data_df.dropna()
    return data_df
    
