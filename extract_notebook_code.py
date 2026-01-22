import json

try:
    with open('sleep_data_analysis.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print("Notebook Cells:")
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            print("\n# Code Cell:")
            print("".join(cell['source']))
        elif cell['cell_type'] == 'markdown':
            print("\n# Markdown Cell:")
            print("".join(cell['source']))
except Exception as e:
    print(f"Error reading notebook: {e}")
