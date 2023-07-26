import json
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
#import mpld3
import pandas as pd
import requests
import io
import datetime
import re
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import uuid, base64
#from keras.models import Sequential
from sklearn.model_selection import learning_curve
#from keras.layers import LSTM, Dense, GRU  # Added import statement for GRU layer

# Set the backend to a non-GUI backend

#matplotlib.use('Agg')

combined_dataset = None
combined_dataset_filtered = None

def is_valid_url(url):
    # Simple URL validation using regular expression
    pattern = re.compile(r'^https?://\S+$', re.IGNORECASE)
    return bool(url and re.match(pattern, url))

def preprocess_dataset(dataset):
    dataset.replace('', pd.NA, inplace=True)

    try:
        dataset = pd.read_csv(io.StringIO(dataset.to_csv(index=False)), encoding='latin1')
    except pd.errors.ParserError:
        print("Error: Invalid file path or file not found.")
        print("Trying an alternative approach to read the data...")
        try:
            dataset = pd.read_csv(io.StringIO(dataset.to_csv(index=False, header=False)), encoding='latin1')
            print("Alternative approach successful.")
        except pd.errors.ParserError:
            print("Error: Unable to read the data using the alternative approach.")
            print("Please check the data format and structure.")

    return dataset

def get_combined_dataset(request):
    target_station_sites = request.GET.get('target_station_sites')
    input_station_sites_list = request.GET.get('input_station_sites_list')
    parameters = request.GET.get('params')

    try:
        combined_datasets = []
        tmin = "01/10/1973"
        tmax = datetime.datetime.today().strftime('%d/%m/%Y')

        # Fetch the target station dataset
        target_station_csv_url = f"https://snirh.apambiente.pt/snirh/_dadosbase/site/paraCSV/dados_csv.php?sites={target_station_sites},{input_station_sites_list}&pars={parameters}&tmin={tmin}&tmax={tmax}&formato=csv"

        r = requests.get(target_station_csv_url)

        if r.status_code == 200:
            output = r.text
            outputsplit = output.split('\n')
            target_station_dataset = pd.DataFrame([o.split(',') for o in outputsplit])

            tsd = target_station_dataset.iloc[3:len(target_station_dataset)-3]
            c = []
            for col in tsd.columns:
                if 'FLAG' in str(tsd[col].iloc[0]) or '\r' in str(tsd[col].iloc[0]):
                    c.append(col)
            tsd = tsd.drop(columns=c)
            tsd = tsd[~tsd[0].str.contains(r'\r')]

            combined_datasets.append(tsd)
            global combined_dataset
            combined_dataset = pd.concat(combined_datasets, axis=1, ignore_index=True)
            combined_dataset.columns = ["date"] + list(combined_dataset.columns[1:])

            combined_table = combined_dataset.to_html(index=False)

            return render(request, 'prediction_table.html', {'combined_table': combined_table, 'combined_dataset': combined_dataset})

        else:
            error_message = f"Error: Unable to download the target station dataset from the URL: {target_station_csv_url}"
            return HttpResponse(error_message)

    except ValueError as e:
        error_message = f"Error: Invalid file path or file not found. Details: {str(e)}"
        return HttpResponse(error_message)

def allsteps_preprocessing_data_before(request):
    global combined_dataset_filtered

   
    start_date_str = "1999-10-01 00:00:00"
    end_date_str = "2023-01-01 00:00:00"

    combined_dataset['date'] = pd.to_datetime(combined_dataset['date'])
    combined_dataset_filtered = combined_dataset[
         (combined_dataset['date'] >= pd.to_datetime(start_date_str)) &
        (combined_dataset['date'] <= pd.to_datetime(end_date_str))
            ]

    combined_dataset_filtered.set_index('date', inplace=True)


    before_plots = []
    for column in combined_dataset_filtered.columns:
        plt.figure()
        plt.plot(combined_dataset_filtered[column], label='Before Missing Values')
        plt.title(f"Column: {column} - Before Missing Values")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        graph = base64.b64encode(image_png).decode('utf-8')
        buffer.close()

        before_plots.append(graph)

    combined_dataset_html = combined_dataset_filtered.to_html()

    return render(request, 'prediction.html',
                  {'combined_dataset': combined_dataset_html, 'plots': before_plots})

def multivariate_data(inputs, output, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    if start_index is None:
        start_index = 0

    if end_index is None:
        end_index = len(inputs)

    if history_size is None:
        history_size = 0

    if step is None:
        step = 1

    if target_size is None:
        target_size = 1

    start_index = start_index + history_size

    for i in range(start_index, end_index - target_size):
        if len(inputs) == 0:  # Handle empty inputs list
            indices = range(i, i + history_size, step)
        else:
            indices = range(i - history_size, i, step)
        data.append(inputs[indices])

        if single_step:
            labels.append(output[i + target_size])
        else:
            labels.append(output[i:i + target_size])

    return np.array(data), np.array(labels)


def train_lstm_model(x_train, y_train, neurons, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(x_train.shape[-2:])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model, history

def train_gru_model(x_train, y_train, neurons, epochs, batch_size):
    model = Sequential()
    model.add(GRU(neurons, input_shape=(x_train.shape[-2:])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model, history

def train_mlp_model(x_train, y_train, neurons, epochs, batch_size):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(x_train.shape[-1],)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model, history


def allsteps_preprocessing_data_after(request):
    past_history = int(request.POST.get('past_history')) if request.POST.get('past_history') else None
    n_step = int(request.POST.get('n_step')) if request.POST.get('n_step') else None
    STEP = int(request.POST.get('STEP')) if request.POST.get('STEP') else None
    train_percentage = int(request.POST.get('train_percentage')) if request.POST.get('train_percentage') else None
    filling_type = request.POST.get('filling_type')

    global combined_dataset_filtered

    combined_dataset_filtered = combined_dataset_filtered.copy()

    if filling_type == "interpolate":
        combined_dataset_filtered = combined_dataset_filtered.interpolate()
    elif filling_type == "bfill":
        combined_dataset_filtered = combined_dataset_filtered.bfill()
    elif filling_type == "ffill":
        combined_dataset_filtered = combined_dataset_filtered.ffill()
    elif filling_type == "mean":
        combined_dataset_filtered = combined_dataset_filtered.fillna(combined_dataset_filtered.mean())

    after_plots = []
    #plot_width = 70 # Change this value to adjust the width of the plots
    #plot_height = 40
    for column in combined_dataset_filtered.columns:
        #plt.figure(figsize=(plot_width, plot_height))
        plt.figure()
        plt.plot(combined_dataset_filtered[column], label='After Missing Values')
        plt.title(f"Column: {column} - After Missing Values")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        graph = base64.b64encode(image_png).decode('utf-8')
        buffer.close()

        after_plots.append(graph)


    for column, graph in zip(combined_dataset_filtered.columns, after_plots):
       #plt.figure(figsize=(plot_width, plot_height))
       plt.figure()
       plt.plot(combined_dataset_filtered[column], label='After Missing Values')
       plt.title(f"Column: {column} - After Missing Values")
       plt.xlabel('Time')
       plt.ylabel('Value')
       plt.legend()

    # Add zoom functionality
       plt.xlim(pd.to_datetime('1999-10-01 00:00:00', format='%Y-%m-%d %H:%M:%S'),
             pd.to_datetime('2023-10-01 00:00:00', format='%Y-%m-%d %H:%M:%S'))  
       plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

       buffer = io.BytesIO()
       plt.savefig(buffer, format='png')
       buffer.seek(0)
       image_png = buffer.getvalue()
       graph = base64.b64encode(image_png).decode('utf-8')
       buffer.close()

    after_plots.append(graph)      

    combined_dataset_html = combined_dataset_filtered.to_html()

    return render(request, 'prediction.html',
                  {'combined_dataset': combined_dataset_html, 'plots': after_plots})


def allsteps_preprocessing_data_split(request):
    print(request.body)
    jsonReq = json.loads(request.body.decode('utf-8'))
    print(jsonReq)
    
    past_history = jsonReq['past_history'] 
    n_step = int(jsonReq['n_step']) if jsonReq['n_step'] else None
    STEP = int(jsonReq['STEP']) if jsonReq['STEP'] else None
    train_percentage = int(jsonReq['train_percentage']) if jsonReq['train_percentage'] else None
    #filling_type = jsonReq['filling_type']
    indices1 = jsonReq['indices1']
    indices2 = jsonReq['indices2']
    print("past_history:", past_history)
    print("train_percentage:", train_percentage)
    print("indices1:", indices2)

    if indices1 is not None and indices2 is not None:
        try:
            indices1 = list(map(int, indices1.split(',')))
            indices2 = int(indices2)
        except ValueError:
            # Handle the case when indices1 or indices2 is not a valid integer or comma-separated list
            indices1 = []
            indices2 = 0
            return HttpResponse("Error: indices1 or indices2 is not a valid integer or comma-separated list.")
    else:
        # Handle the case when indices1 or indices2 is not provided
        indices1 = []
        indices2 = 0
        return HttpResponse("Error: indices1 or indices2 is not provided.")

    global combined_dataset_filtered

    if combined_dataset_filtered is None:
        error_message = "Error: combined_dataset_filtered is not defined. Please make sure to preprocess the dataset first."
        return HttpResponse(error_message)

    dataset2 = combined_dataset_filtered.to_numpy()
    if train_percentage is None:
        error_message = "Error: train_percentage is not defined. Please provide a valid train percentage value."
        return HttpResponse(error_message)
     
    split_index = int(len(dataset2) * train_percentage / 100)
    
    #split_index = int(past_history)
    
    x_train_single, y_train_single = multivariate_data(dataset2[:, indices1], dataset2[:, indices2],
                                                  0, split_index, int(past_history), n_step, STEP,
                                                  single_step=True)

    print("1")

    x_test_single, y_test_single = multivariate_data(dataset2[:, indices1], dataset2[:, indices2],
                                                    split_index, None, int(past_history), n_step, STEP,
                                                    single_step=True)
    
    x_train_single = [item[0][0] for item in x_train_single]  # Convert to list
    x_test_single = [item[0][0] for item in x_test_single] 
   

    
   
    plot_preprocessing=[]
    # Plotting train and test datasets
    plt.figure()
    plt.plot(range(len(x_train_single)), x_train_single, label='Train Dataset')
    plt.plot(range(len(x_train_single), len(x_train_single) + len(x_test_single)), x_test_single, label='Test Dataset')
    plt.title('Train and Test Datasets')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    # Plotting train and test datasets
 
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    print("2")

    plot_preprocessing.append(graph)

    x_train_single_df = pd.DataFrame({'Value': x_train_single})
    x_train_single_html = x_train_single_df.to_html(index=False)
    print("3")



    # try: 
    #     response_data = {
    #         'x_train_single': x_train_single,
    #         'x_test_single': x_test_single,
    #         # Include other relevant data in the response
    #     }


    #     # Return the JSON response
    #     return JsonResponse({'data': response_data})
    # except Exception as e:
    #     # Handle any exceptions and return an error JSON response
    #     error_message = str(e)
        # return JsonResponse({'error': error_message})
    
    return render(request, 'prediction.html', {'dataset2': dataset2, 'plots': plot_preprocessing})

   




def index(request):
    return render(request, 'index.html')

def prediction_form(request):
    if request.method == 'POST':
        return get_combined_dataset(request)
    return render(request, 'prediction_form.html')
