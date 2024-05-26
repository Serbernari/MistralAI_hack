import re
import streamlit as st
import pandas as pd
from io import StringIO
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import concurrent.futures

# Initialize Mistral Client
api_key = "dnDSPYh1mNwZyGRSQgjCPKQc144MfMng"
model = "mistral-large-latest"
client = MistralClient(api_key=api_key)

def get_mistral_convert_units(item):
    """Converts item units to the International System of Units (SI)."""
    chat_response = client.chat(
        model=model,
        messages=[
            ChatMessage(
                role="user",
                content=f'Please convert this unit into the International System of Units: "{item}". Provide a short answer.'
            )
        ]
    )
    res1 = chat_response.choices[0].message.content
    
    chat_response2 = client.chat(
        model=model,
        messages=[
            ChatMessage(role="system", content='Extract number and SI unit from the given text. Format your text like "10 kilograms" or "1 litre".'),
            ChatMessage(role="user", content=f'text:{res1}')
        ]
    )
    res2 = chat_response2.choices[0].message.content
    
    chat_response3 = client.chat(
        model=model,
        messages=[
            ChatMessage(role="system", content='Extract number and SI unit from the given text. Format your text like "10 kilograms" or "1 litre". Answer only with number and unit, nothing else.'),
            ChatMessage(role="user", content=f'text:{res2}')
        ]
    )
    res3 = chat_response3.choices[0].message.content
    
    return res3

def mistral_compare(text):
    """Compares shopping list items."""
    chat_response = client.chat(
        model=model,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    'You will be asked questions about items to determine if they are the same name for a product inside grocery store. '
                    'Specifically, you will need to assess if item_A and item_B are the same in terms of being found in the same section or location within the store. '
                    'For example, "картошка" and "картофелина" should be considered the same since they can be bought in the same place, and you should answer "yes". '
                    'Please answer with "yes" or "no".'
                )
            ),
            ChatMessage(role="user", content=f'{text}')
        ]
    )
    return chat_response.choices[0].message.content

def mistral_compare_items(data_df):
    """Compares items in a DataFrame and merges similar items."""
    input_list = []
    index_pairs = []
    n = len(data_df)

    for i in range(n):
        for j in range(i + 1, n):  # Compare each pair only once
            item_A = data_df.loc[i, 'Item']
            item_B = data_df.loc[j, 'Item']
            if item_A is not None and item_B is not None:
                input_list.append((item_A, item_B))
                index_pairs.append((i, j))

    texts = [f'Are "{pair[0]}" and "{pair[1]}" the same?' for pair in input_list]
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    total_tasks = len(texts)

    def update_progress(future):
        progress_bar.progress((futures.index(future) + 1) / total_tasks)
        
    # Perform parallel API calls with progress update
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(mistral_compare, text) for text in texts]
        for future in concurrent.futures.as_completed(futures):
            update_progress(future)
            # Collect the result to ensure all futures are processed
            future.result()
            
    # Collect the results after all tasks are completed
    results = [future.result() for future in futures]
    progress_bar.empty()
    # Create a merge dictionary to keep track of items to merge
    merge_dict = {}
    for k, result in enumerate(results):
        i, j = index_pairs[k]
        if "yes" in result.lower():
            if i in merge_dict:
                merge_dict[i].append(j)
            else:
                merge_dict[i] = [j]

    # Merge the items based on the comparison results
    for i, merge_indices in merge_dict.items():
        for j in merge_indices:
            if data_df.loc[j, 'Item'] is not None:  # Only merge if item is still present
                data_df.loc[i, 'Amount'] += data_df.loc[j, 'Amount']
                data_df.loc[j, 'Item'] = None

    # Remove rows where 'Item' is None
    return data_df.dropna(subset=['Item'])

def mistral_create_csv(input):
    """Creates a CSV from the input text."""
    chat_response = client.chat(
        model=model,
        messages=[ChatMessage(
            role="user",
            content=(
                f'Organize this shopping list in a CSV file with columns: Item, Amount, Unit. '
                f'Write it so I can open it in pandas later, so no additional text or comments, never. {input}'
            )
        )]
    )
    return chat_response.choices[0].message.content

def extract_first_number(text):
    """Extracts the first number from the given text, considering both integers and floats."""
    match = re.search(r'\b\d+(\.\d+)?', text)
    if match:
        return float(match.group(0))
    return 1

def create_csv(input_text):
    """Converts the CSV string to a pandas DataFrame."""
    csv_content = mistral_create_csv(input_text)
    string_data = StringIO(csv_content)
    df = pd.read_csv(string_data)
    
    df['Amount'] = df['Amount'].fillna('1').astype(str).apply(extract_first_number)
    df['Unit'] = df['Unit'].fillna('pcs')
    df['Done'] = False
    
    return df

