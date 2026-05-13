import pandas as pd

def save_data(data, filename, save_path, txt_path=False):
    """
    Save data from the raw files into csv files

    Parameters:
    - data: data to save
    - filename : name of the file
    - save_path : where to save the data  
    """
    df = pd.DataFrame(data, index = range(0, len(data["depth"])))
    df.to_csv(f'{save_path}/{filename}')

    if txt_path:
        with open(txt_path, 'a') as txt_doc:
            txt_doc.write(f'{filename}\n')