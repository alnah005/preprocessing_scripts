# -*- coding: utf-8 -*-
"""
file: lines_analysis.py

@author: Suhail.Alnahari

@description: This file generates all_lines.csv and all_boxes.csv

@created: 2021-08-31T11:32:25.805Z-05:00

@last-modified: 2021-11-01T13:19:52.331Z-05:00
"""

# standard library
# 3rd party packages
# local source
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan

## only removes keywords that have a closing tag
def removeSquareBracketKeywords(text):
    keywordsFound = []
    current_keyword = ""
    openBracket = False
    for i in text:
        if i == "[" and not(openBracket):
            openBracket = True
            continue
        elif i == "]" and openBracket:
            openBracket = False
            keywordsFound.append(current_keyword)
            current_keyword = ""
            continue
        if openBracket:
            current_keyword += i
    for key in keywordsFound:
        if "["+key+"]" in text and "[/"+key+"]" in text:
            text = text.replace("["+key+"]","")
            text = text.replace("[/"+key+"]","")
    return text


labels = pd.read_csv("poly_line_text_reducer_htr.csv")

frame_to_points = {
    "subjectId_frame_classification":[],
    "x_0": [], "y_0": [],
    "x_1": [], "y_1": [],
    "consensus_text":[]
}

for index, row in labels.iterrows():
    subject_id = row['subject_id']
    for k in row.iteritems():
        if "data.frame" in k[0] and not(pd.isnull(k[1])):
            frameNum = k[0].split('data.frame')[-1]
            try:
                for i,dic in enumerate(list(eval(k[1]))):
                    x_0,y_0,x_1,y_1 = (dic['clusters_x'][0],dic['clusters_y'][0],dic['clusters_x'][1],dic['clusters_y'][1])
                    frame_to_points['subjectId_frame_classification'].append(f"{subject_id}_{frameNum}_{i}")
                    frame_to_points['x_0'].append(x_0)
                    frame_to_points['y_0'].append(y_0)
                    frame_to_points['x_1'].append(x_1)
                    frame_to_points['y_1'].append(y_1)
                    frame_to_points["consensus_text"].append(dic['consensus_text'])
            except:
                print(k[1])

pd.DataFrame.from_dict(frame_to_points).to_csv("all_lines.csv",index=False)

lowerQuantile = 0.1
upperQuantile = 0.2
all_lines = pd.read_csv("all_lines.csv")

# from https://math.stackexchange.com/questions/2043054/find-a-point-on-a-perpendicular-line-a-given-distance-from-another-point


@dataclass
class Classification:
    classificationOrder: int
    x_0: int
    y_0: int
    x_1: int
    y_1: int
    text: str
    max_diff: Optional[float] = None

    def __post_init__(self):
        super().__init__()
        assert isinstance(self.classificationOrder, int)
        self.text = removeSquareBracketKeywords(self.text)


@dataclass
class Subject:
    id: str
    classifications: List[Classification]

    def _distance(self, c1: Classification, c2: Classification, initial=False):
        if initial:
            return (
                abs((c1.y_0 - c2.y_0) / max(c2.y_0, c1.y_0, 1)) ** 2
                + abs((c1.x_0 - c2.x_0) / max(c2.y_0, c1.y_0, 1)) ** 2
            ) ** 0.5
        return (
            abs((c1.y_1 - c2.y_1) / max(c2.y_1, c1.y_1, 1)) ** 2
            + abs((c1.x_1 - c2.x_1) / max(c2.x_1, c1.x_1, 1)) ** 2
        ) ** 0.5

    def closestMatch(self, c, left, initial=False):
        distances = {
            index: self._distance(self.classifications[c], l, initial=initial)
            for index, l in enumerate(left)
            if l.classificationOrder != self.classifications[c].classificationOrder
        }
        if len(distances) == 0 and len(left) == 1:
            result = left[0]
            del left[0]
            return result
        result = left[min(distances, key=lambda k: distances[k])]
        del left[min(distances, key=lambda k: distances[k])]
        return result

    def getClassificationSortedByY0(self, reverse=False):
        result = []
        left = self.classifications.copy()
        for c in range(len(self.classifications)):
            result.append(self.closestMatch(c, left, initial=True))
        if reverse:
            return result[::-1]
        return result

    def getClassificationSortedByY1(self, reverse=False):
        result = []
        left = self.classifications.copy()
        for c in range(len(self.classifications)):
            result.append(self.closestMatch(c, left))
        if reverse:
            return result[::-1]
        return result

    def maximumDifference(self):
        y0Classifications = self.getClassificationSortedByY0()
        y1Classifications = self.getClassificationSortedByY1()
        differences = {}
        for c in range(len(y0Classifications) - 1):
            differences[y0Classifications[c].classificationOrder] = abs(
                y0Classifications[c].y_0 - y0Classifications[c + 1].y_0
            )
        for c in range(len(y1Classifications) - 1):
            diff = abs(y1Classifications[c].y_1 - y1Classifications[c + 1].y_1)
            if differences.get(y1Classifications[c].classificationOrder, None) is None:
                differences[y1Classifications[c].classificationOrder] = diff
            else:
                differences[y1Classifications[c].classificationOrder] = max(
                    diff, differences[y1Classifications[c].classificationOrder]
                )
        for c in range(len(self.classifications)):
            if self.classifications[c].classificationOrder in differences:
                self.classifications[c].max_diff = differences[
                    self.classifications[c].classificationOrder
                ]
        return differences


subjects: Dict[str, Subject] = {}
for index, row in all_lines.iterrows():
    subjectId, frame, classificationOrder = row["subjectId_frame_classification"].split(
        "_"
    )
    x_0, y_0, x_1, y_1 = row["x_0"], row["y_0"], row["x_1"], row["y_1"]
    if subjects.get(f"{subjectId}_{frame}", None) is None:
        subjects[f"{subjectId}_{frame}"] = Subject(f"{subjectId}_{frame}", [])
    subjects[f"{subjectId}_{frame}"].classifications.append(
        Classification(
            classificationOrder=int(classificationOrder),
            x_0=x_0,
            y_0=y_0,
            x_1=x_1,
            y_1=y_1,
            text=row["consensus_text"],
        )
    )


differences: Dict[str, Dict[str, float]] = {}
for k, v in subjects.items():
    differences[k] = v.maximumDifference()

differencesNormalized: List[float] = []
differencesUnNormalized: List[float] = []
for k, diff in differences.items():
    differencesNormalized.append(sum(diff.values()) / max(len(diff), 1))
    differencesUnNormalized += list(diff.values())

graphics = False
if graphics:
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    x = np.asarray(differencesNormalized)
    pd_series = pd.Series(x)
    pd_series_adjusted = pd_series[
        pd_series.between(
            pd_series.quantile(lowerQuantile), pd_series.quantile(upperQuantile)
        )
    ]

    ax1.boxplot(pd_series, whis=[round(lowerQuantile * 10), round(upperQuantile * 10)])
    ax1.set_title("Original Normalized")

    ax2.boxplot(
        pd_series_adjusted, whis=[round(lowerQuantile * 10), round(upperQuantile * 10)]
    )
    ax2.set_title("Adjusted Normalized")

    x = np.asarray(differencesUnNormalized)
    pd_series = pd.Series(x)
    pd_series_adjusted = pd_series[
        pd_series.between(
            pd_series.quantile(lowerQuantile), pd_series.quantile(upperQuantile)
        )
    ]
    ax3.boxplot(pd_series, whis=[round(lowerQuantile * 10), round(upperQuantile * 10)])
    ax3.set_title("Original UnNormalized")

    ax4.boxplot(
        pd_series_adjusted, whis=[round(lowerQuantile * 10), round(upperQuantile * 10)]
    )
    ax4.set_title("Adjusted UnNormalized")

    plt.show()

x = np.asarray(differencesNormalized)
pd_series = pd.Series(x)
pd_series_adjusted = pd_series[
    pd_series.between(
        pd_series.quantile(lowerQuantile), pd_series.quantile(upperQuantile)
    )
]

for k, v in subjects.items():
    for classification in v.classifications:
        randomHeight = np.random.choice(pd_series_adjusted)
        if classification.max_diff is None or classification.max_diff > randomHeight:
            classification.max_diff = randomHeight


def lin_equ(l1, l2):
    """Line encoded as l=(x,y)."""
    if abs(l2[0] - l1[0]) < 0.000001:
        return 100000000, -10000000
    m = ((l2[1] - l1[1])) / (l2[0] - l1[0])
    c = l2[1] - (m * l2[0])
    return m, c


@dataclass
class ClassificationWithBox:
    subjectId_frame_classification: str
    x_0: float
    y_0: float
    x_1: float
    y_1: float
    text: str
    height: Optional[float]
    x_2: Optional[float] = None
    y_2: Optional[float] = None
    x_3: Optional[float] = None
    y_3: Optional[float] = None

    def __post_init__(self):
        super().__init__()
        assert not (self.height is None)

    def computeBoxes(self):
        m, c = lin_equ((self.x_1, self.y_1), (self.x_0, self.y_0))
        result = 0
        if m != 0:
            if m == 100000000:
                self.x_2 = self.x_0 + self.height
                self.x_3 = self.x_1 + self.height
                self.x_0 = self.x_0 - self.height
                self.x_1 = self.x_1 - self.height
                self.y_2 = self.y_0
                self.y_3 = self.y_1
                result = 1
            else:
                self.y_2 = self.y_0 - ((self.height ** 2) / (m ** 2 + 1)) ** 0.5
                self.y_3 = self.y_1 - ((self.height ** 2) / (m ** 2 + 1)) ** 0.5
                self.x_2 = self.x_0 + (self.height * m * ((m ** 2 + 1) ** -1) ** 0.5)
                self.x_3 = self.x_1 + (self.height * m * ((m ** 2 + 1) ** -1) ** 0.5)
        else:
            self.x_2 = self.x_0
            self.x_3 = self.x_1
            self.y_2 = self.y_0 + self.height
            self.y_3 = self.y_1 + self.height
            self.y_0 = self.y_0 - self.height
            self.y_1 = self.y_1 - self.height
            result = 1
        self.x_2 = max(self.x_2, 0)
        self.x_3 = max(self.x_3, 0)
        self.y_2 = max(self.y_2, 0)
        self.y_3 = max(self.y_3, 0)
        self.x_0 = max(self.x_0, 0)
        self.x_1 = max(self.x_1, 0)
        self.y_0 = max(self.y_0, 0)
        self.y_1 = max(self.y_1, 0)
        return result


boxedClassifications = {}
for k, v in subjects.items():
    subject_frame = v.id
    for classification in v.classifications:
        boxedClassifications[
            f"{subject_frame}_{classification.classificationOrder}"
        ] = ClassificationWithBox(
            subjectId_frame_classification=f"{subject_frame}_{classification.classificationOrder}",
            x_0=classification.x_0,
            y_0=classification.y_0,
            x_1=classification.x_1,
            y_1=classification.y_1,
            height=classification.max_diff,
            text=classification.text,
        )

errors = 0
for k, b in boxedClassifications.items():
    errors += b.computeBoxes()
print("Number of perfectly horizontal or vertical lines = " + str(errors))

final_boxes: Dict[str, List] = {
    "subjectId_frame_classification": [],
    "x_0": [],
    "y_0": [],
    "x_1": [],
    "y_1": [],
    "height": [],
    "x_2": [],
    "y_2": [],
    "x_3": [],
    "y_3": [],
    "text": [],
}
for k, b in boxedClassifications.items():
    final_boxes["subjectId_frame_classification"].append(
        b.subjectId_frame_classification
    )
    final_boxes["x_0"].append(b.x_0)
    final_boxes["y_0"].append(b.y_0)
    final_boxes["x_1"].append(b.x_1)
    final_boxes["y_1"].append(b.y_1)
    final_boxes["x_2"].append(b.x_2)
    final_boxes["y_2"].append(b.y_2)
    final_boxes["x_3"].append(b.x_3)
    final_boxes["y_3"].append(b.y_3)
    final_boxes["height"].append(b.height)
    final_boxes["text"].append(b.text)

pd.DataFrame.from_dict(final_boxes).to_csv("all_boxes.csv", index=False)

print("You must threshold the boxes to match the dimensions of individual images")

