import pandas as pd
from collections import defaultdict
import os
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import time
count = 1

def readImg(url, grey=True):
    global count
    if (count+1) % 5000 == 0:
        time.sleep(60)
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.convert("L")
        return img
    except:
        print(url)
    return None
subj_to_links = pd.read_csv("/home/fortson/alnah005/aggregation_for_caesar/htr_gold_standard/Initial_10_from10pct.csv")
subj_to_loc_images = defaultdict(list)
dataLocation = "/home/fortson/alnah005/text_recognition/data/umnDatasetSample/"

howManyDownloaded = 0
id_to_name = {'id':[],'name':[]}
for index, row in tqdm(subj_to_links.iterrows()):
    subj_id = row['id']
    fileLinks = [(i,row[f"file_{i}"]) for i in range(len([None for f in row.keys() if "file" in f])) if row[f"file_{i}"] != "None"]
    fileNames = [(i,link.split('/')[-1],link) for i,link in fileLinks]
    for frame,name,link in fileNames:
        if not(os.path.isfile(f"{dataLocation}{name}")):
            img = readImg(link)
            if img is None:
                continue
            img.save(f"{dataLocation}{name}")
            howManyDownloaded += 1
            count += 1
        id_to_name['id'].append(str(subj_id)+"_"+str(frame))
        id_to_name['name'].append(f"{dataLocation}{name}")
print(howManyDownloaded)
id_to_name_pd = pd.DataFrame.from_dict(id_to_name)
id_to_name_pd.to_csv("/home/fortson/alnah005/aggregation_for_caesar/htr_gold_standard/subject_to_name.csv",index=False)



