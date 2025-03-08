import requests
import os
from io import BytesIO
import csv
from tqdm import tqdm

# PhysioNet credentials
username = "jpthesmar"  # Replace with your PhysioNet username
password = "Bebebaby123.."  # Replace with your PhysioNet password

# Azure Storage credentials
connection_string = "DefaultEndpointsProtocol=https;AccountName=sleepdataset;AccountKey=PMH2yxMI11CX433WKWpxyRuJ/96H4susL+n7FIdg+qzXd0qtcy1q3/QHzX/qC+5Jk7xzW7H3r4ta+ASts5TK/A==;EndpointSuffix=core.windows.net"  # Replace with your Azure connection string
container_name = "dreamt-dataset"  # Replace with your container name

# PhysioNet authentication session
session = requests.Session()
auth_url = "https://physionet.org/login/"

# First, get the CSRF token
response = session.get(auth_url)
if response.status_code != 200:
    raise Exception("Failed to access PhysioNet login page")

# Extract CSRF token from the response
csrf_token = None
for line in response.text.split('\n'):
    if 'csrfmiddlewaretoken' in line:
        # Basic extraction, might need refinement depending on page structure
        csrf_start = line.find('value="') + 7
        csrf_end = line.find('"', csrf_start)
        csrf_token = line[csrf_start:csrf_end]
        break

if not csrf_token:
    raise Exception("Could not find CSRF token")

# Login to PhysioNet
login_data = {
    'username': username,
    'password': password,
    'csrfmiddlewaretoken': csrf_token,
    'next': '/'
}

response = session.post(auth_url, data=login_data, headers={
    'Referer': auth_url
})

if response.status_code != 200:
    raise Exception(f"Login failed with status code {response.status_code}")
else:
    print('Login Successful')

def download_file(file_url, local_path):
    """Download a file from PhysioNet to local storage with progress bar"""
    # Check if file already exists and is complete
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        response = session.head(file_url)
        remote_size = int(response.headers.get('content-length', 0))
        
        if local_size == remote_size:
            print(f"File already exists and is complete: {local_path}")
            return True
        else:
            print(f"File exists but is incomplete. Redownloading: {local_path}")
            os.remove(local_path)
    
    # Stream download with progress bar
    with session.get(file_url, stream=True) as response:
        if response.status_code != 200:
            print(f"Failed to download {file_url}. Status code: {response.status_code}")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_path)) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True


# Example usage - transfer one file

local_dir = "data/S008_whole_df.csv"
file_url = "https://physionet.org/files/dreamt/2.0.0/data_64Hz/S008_whole_df.csv" 
local_file = "S008_whole_df.csv"

files = [f"S{str(i).zfill(3)}_whole_df.csv" for i in range(8, 103)]
for file in files[:23]:
    file_url = f'https://physionet.org/files/dreamt/2.0.0/data_64Hz/{file}'
    local_dir = f'data/{file}'
    download_file(file_url, local_dir)