import os
import json
import numpy as np
from scipy.stats import multivariate_normal
import warnings

training_json_path = "final_annotations_10_train_cropped.json"
training_json_output_path = "final_annotations_10_train_cropped_stats_filled.json"
training_json_error_output_path = "final_annotations_10_train_cropped_stats_filled_errors.json"
prediction_json_path = "detections_cropped_with_text.json"
prediction_json_output_path = "detections_cropped_with_text_stats_filled.json"
prediction_json_error_output_path = "detections_cropped_with_text_stats_filled_errors.json"
histogram_cutoff = 0.000245
entropyCutoff = 0.1
enable_graphs = False
if enable_graphs:
    import matplotlib.pyplot as plt
assert os.path.isfile(prediction_json_path)
with open(prediction_json_path, 'r') as openfile:
    detections = json.load(openfile)

assert os.path.isfile(training_json_path)
with open(training_json_path, 'r') as openfile:
    training = json.load(openfile)

def getWordEntropy(word):
    freq = {k:0 for i in word.lower().split(' ') for k in i if i != ' '}
    for i in word.lower().split(' '):
        for k in i:
            if k != ' ':
                freq[k] += 1
    if sum(freq.values()) == 0 or len(freq) == 0:
        return 0
    if len(freq) == 1 and sum(freq.values()) == 1:
        return 1
    return -1*sum([v*np.log(v/sum(freq.values()))/sum(freq.values()) for v in freq.values()])/len(freq)

def getSlope(x,y):
    slope_degree, slope = -np.inf, -90
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = np.polyfit(x, y, 1)
            y_fit = np.polyval(fit, [x[0], x[-1]])
            dx = x[-1] - x[0]
            dy = y_fit[-1] - y_fit[0]
            slope_degree = np.rad2deg(np.arctan2(dy, dx))
            slope = dy/dx if dx != 0 else -np.inf
        except np.RankWarning:
            try:
                # rotate by 90 before fitting
                x_tmp = -np.array(y)
                y_tmp = np.array(x)
                fit = np.polyfit(x_tmp, y_tmp, 1)
                y_fit = np.polyval(fit, [x_tmp[0], x_tmp[-1]])
                dx = x_tmp[-1] - x_tmp[0]
                dy = y_fit[-1] - y_fit[0]
                if dy.round(6) == 0:
                    # convert -0 into 0 so `arctan2` gives the expect results for horizontal lines
                    dy = 0.0
                # rotate by -90 to bring back into correct coordinates
                slope_degree = np.rad2deg(np.arctan2(dy, dx)) - 90
                slope = dy/dx if dx.round(6) != 0 else -np.inf
            except np.RankWarning:
                # this is the case where dx = dy = 0 (a line of zero length)
                slope, slope_degree = 0.0, 0.0
        except:
            if np.isnan(slope) or np.isnan(slope_degree):
                return -np.inf, 90
    return slope,slope_degree

def getUnderlineWithSlope(x_p,y_p,w_p,h_p,a_p,radians=False):
    cx = (2*x_p+w_p)/2
    cy = (2*y_p+h_p)/2
    centre = np.array([cx, cy])
    original_points = np.array(
        [
        [cx - 0.5 * w_p, cy - 0.5 * h_p],  # This would be the box if theta = 0
        [cx + 0.5 * w_p, cy - 0.5 * h_p],
        [cx + 0.5 * w_p, cy + 0.5 * h_p],
        [cx - 0.5 * w_p, cy + 0.5 * h_p],
        ]
    )
    if not(radians):
        a_p = a_p* np.pi/180 
    rotation = np.array([[np.cos(a_p), np.sin(a_p)], [-np.sin(a_p), np.cos(a_p)]])
    corners = np.matmul(original_points - centre, rotation) + centre
    corners = corners.astype(int)
    max_index = -1
    max_y = 0
    for index,i in enumerate(corners):
            x,y = i
            if y > max_y:
                max_y = y
                max_index = index
    points1 = [[corners[max_index][0],corners[(max_index+1)%len(corners)][0]],[corners[max_index][1],corners[(max_index+1)%len(corners)][1]]]
    points2 = [[corners[max_index][0],corners[(max_index-1+len(corners))%len(corners)][0]],[corners[max_index][1],corners[(max_index-1+len(corners))%len(corners)][1]]]

    slope1,deg1 = getSlope(*points1)
    slope2,deg2 = getSlope(*points2)
    if abs(slope1) > abs(slope2):
        return [[int(p) for p in axis] for axis in points2],deg2, slope2
    return [[int(p) for p in axis] for axis in points1],deg1, slope1


for index,i in enumerate(training):
    training[index]["crop_w_div_h"] = i["crop_w"]/i["crop_h"] if (i["crop_w"]/i["crop_h"]) > 1 else (i["crop_h"]/i["crop_w"])
    training[index]["text_len"] = len([l for l in i["text"] if l != " "])
    points, slope_deg, slope = getUnderlineWithSlope(*training[index]['bbox'],radians = True)
    training[index]["x"], training[index]["y"] = points
    training[index]["slope"] = float(slope)
    training[index]["slope_deg"] = float(slope_deg)

# Creating plot
x = [i["crop_w_div_h"] for i in training]
y = [i["text_len"] for i in training]
x_filtered = [i["crop_w_div_h"] for i in training if i["text_len"] < 100 and  i["crop_w_div_h"] < 20]
y_filtered = [i["text_len"] for i in training if i["text_len"] < 100 and  i["crop_w_div_h"] < 20]
data = np.asarray([x_filtered,y_filtered])
x_min = np.min(x_filtered)
x_max = np.max(x_filtered)
  
y_min = np.min(y_filtered)
y_max = np.max(y_filtered)
  
x_bins = np.linspace(x_min, x_max, 500)
y_bins = np.linspace(y_min, y_max, 500)
if enable_graphs:
    fig, ax = plt.subplots(figsize =(10, 7))
    # Creating plot
    plt.hist2d(x_filtered, y_filtered, bins =[x_bins, y_bins])
    plt.title("2d histogram of sequence length vs aspect ratio of image")
    
    ax.set_xlabel('width/height') 
    ax.set_ylabel('Num of characters') 
    
    # show plot
    plt.tight_layout() 
    plt.show()

mesh_x,mesh_y = np.meshgrid(x_bins, y_bins)
mean = np.mean(data,axis=1)
cov = np.cov(data)
var = multivariate_normal(mean=mean, cov=cov)
pos = np.dstack((mesh_x, mesh_y))

if enable_graphs:
    plt.contourf(mesh_x, mesh_y, var.pdf(pos))
    plt.show() 

    plt.hist(var.pdf(pos))
    plt.show()

    plt.contourf(mesh_x, mesh_y, (var.pdf(pos) > histogram_cutoff).astype(int))
    plt.show() 

for index,i in enumerate(training):
    training[index]["gaussian_fit_score"] = var.pdf([training[index]["crop_w_div_h"],training[index]["text_len"]])
    training[index]["sentence_avg_entropy"] = getWordEntropy(i["text"])

if enable_graphs:
    plt.hist([i["sentence_avg_entropy"] for i in training])
    plt.show()

for index,i in enumerate(detections):
    detections[index]["crop_w_div_h"] = i["crop_w"]/i["crop_h"] if (i["crop_w"]/i["crop_h"]) > 1 else (i["crop_h"]/i["crop_w"])
    detections[index]["text_len"] = len([l for l in i["text"] if l != " "])
    detections[index]["gaussian_fit_score"] = var.pdf([detections[index]["crop_w_div_h"],detections[index]["text_len"]])
    detections[index]["sentence_avg_entropy"] = getWordEntropy(i["text"])
    points, slope_deg, slope = getUnderlineWithSlope(*detections[index]['bbox'],radians = True)
    detections[index]["x"], detections[index]["y"] = points
    detections[index]["slope"] = float(slope)
    detections[index]["slope_deg"] = float(slope_deg)

if enable_graphs:
    plt.hist([i["sentence_avg_entropy"] for i in detections])
    plt.show()

print("Make sure you check the histogram created by this script to identify the correct cutoff.")
print("enabled_graphs?:",enable_graphs)
print("current cutoff = ", histogram_cutoff)
print("current recommended entropy score = ", entropyCutoff)


def filterData(i,hist_cutoff=histogram_cutoff,box_cutoff=0.5,entropy_cut=entropyCutoff,len_cutoff=100):
    return i["gaussian_fit_score"] >= hist_cutoff and i["text_len"] < len_cutoff and (i["box_score"] is None or i["box_score"]> box_cutoff) and i["sentence_avg_entropy"] > entropy_cut and abs(i["slope"]) < 0.2


valid_preds = []
invalid_preds = []
for i in detections:
    if filterData(i):
        valid_preds.append(i)
    else:
        invalid_preds.append(i)

with open(prediction_json_output_path, "w") as outfile:
    json.dump(valid_preds,outfile,indent=3)

with open(prediction_json_error_output_path, "w") as outfile:
    json.dump(invalid_preds,outfile,indent=3)

valid_train = []
invalid_train = []
for i in training:
    if filterData(i):
        valid_train.append(i)
    else:
        invalid_train.append(i)

with open(training_json_output_path, "w") as outfile:
    json.dump(valid_train,outfile,indent=3)

with open(training_json_error_output_path, "w") as outfile:
    json.dump(invalid_train,outfile,indent=3)


print("num of detections filtered from train compared to kept",len(invalid_train),":", len(valid_train))
print("num of detections filtered from preds compared to kept",len(invalid_preds),":", len(valid_preds))