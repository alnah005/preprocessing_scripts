from dataclasses import dataclass, fields
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from abc import ABC
from shapely.geometry import Polygon, Point as PolygonPoint
import cv2
import numpy as np


def getCornersFromRotatedBox(
    x_p, y_p, w_p, h_p, a_p, radians=False
) -> List[List[float]]:
    cx = (2 * x_p + w_p) / 2
    cy = (2 * y_p + h_p) / 2
    centre = np.array([cx, cy])
    original_points = np.array(
        [
            [cx - 0.5 * w_p, cy - 0.5 * h_p],  # This would be the box if theta = 0
            [cx + 0.5 * w_p, cy - 0.5 * h_p],
            [cx + 0.5 * w_p, cy + 0.5 * h_p],
            [cx - 0.5 * w_p, cy + 0.5 * h_p],
        ]
    )
    if not (radians):
        a_p = a_p * np.pi / 180
    rotation = np.array([[np.cos(a_p), np.sin(a_p)], [-np.sin(a_p), np.cos(a_p)]])
    corners = np.matmul(original_points - centre, rotation) + centre
    return corners.astype(int).tolist()


def getRotatedBoxFromCorners(corners: List[List[int]], radians=False) -> List[float]:
    rect = cv2.minAreaRect(np.array(corners))
    # for axis aligned
    cx: float = rect[0][0]
    cy: float = rect[0][1]
    w: float = rect[1][0]
    h: float = rect[1][1]
    a: float = rect[2]
    if radians:
        a = a * np.pi / 180
    return [
        max(int(round(cx - 0.5 * w)), 0),
        max(int(round(cy - 0.5 * h)), 0),
        int(round(w)),
        int(round(h)),
        a,
    ]


def getFrameNumberAndFrequency(
    objects: List["Extractable"],
) -> Tuple[int, Dict[int, int]]:
    assert len(objects) > 0
    freqs: Dict[int, int] = {o.frame: 0 for o in objects}
    for o in objects:
        freqs[o.frame] += 1

    frame: int = max(freqs, key=freqs.get)
    return frame, freqs


class PointClass(ABC):
    @staticmethod
    def distance(x_1, y_1, x_2, y_2):
        return (abs(x_1 - x_2) ** 2 + abs(y_1 - y_2) ** 2) ** 0.5

    @property
    def points(self) -> List[int]:
        pass

    def interaction(self, object: "Extractable"):
        pass


class PolygonClass(ABC):
    @staticmethod
    def iou(points1: List[List[int]], points2: List[List[int]]):
        poly_1 = Polygon(points1)
        poly_2 = Polygon(points2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou

    @staticmethod
    def distance(points1: List[List[int]], points2: List[List[int]]):
        return 1 - PolygonClass.iou(points1, points2)

    @property
    def points(self) -> List[List[int]]:
        pass

    def interaction(self, object: "Extractable"):
        pass


class Cluster(ABC):
    pass


@dataclass
class Extractable(ABC):
    id: int
    list_order: int
    frame: int
    task: str
    subject_id: int
    tool: str

    @classmethod
    def from_json(cls, json: Dict[str, Any]):
        class_fields = [f.name for f in fields(cls)]
        non_na_fields = {
            k: v
            for k, v in json.items()
            if isinstance(v, (list, dict)) or not (pd.isna(v))
        }
        for c in class_fields:
            assert c in json
        ### this will fail if a class is non nullable and a null value was passed in
        constructor = {
            k: v for k, v in json.items() if k in non_na_fields if k in class_fields
        }
        result = None
        try:
            result = cls(**constructor)
        except:
            pass
        assert result is not None
        return result

    @classmethod
    def field_names(cls):
        return [field.name for field in fields(cls)]

    @staticmethod
    def getFieldListFromListOfExtractables(objects: List["Extractable"]):
        all_fields = {
            f: []
            for o in objects
            for f in o.field_names()
            if f not in o.extract_defining_columns()
        }
        for o in objects:
            for k in all_fields:
                all_fields[k].append(o.__getattribute__(k))
        return all_fields

    @classmethod
    def extract_defining_columns(cls):
        pass

    def distance_between_classes(self, object: "Extractable") -> float:
        if isinstance(self, (PolygonClass, PointClass)):
            return self.interaction(object)
        raise NotImplementedError()

    def toCluster(self):
        pass

    @classmethod
    def clusterLabels(
        cls, objects: List["Extractable"], distances: List[float], threshold=70
    ) -> Dict[str, Any]:
        pass


@dataclass
class Point(Extractable, PointClass):
    x: float
    y: float

    @property
    def points(self):
        return [self.x, self.y]

    @classmethod
    def extract_defining_columns(cls):
        return ["x", "y"]

    @classmethod
    def clusterLabels(
        cls, objects: List["Point"], distances: List[float], threshold=70
    ) -> Dict[str, Any]:
        assert len(objects) > 0
        assert len(objects) == len(distances)
        non_null_distance_objects = [
            o for o, d in zip(objects, distances) if d is not None and d < threshold
        ]
        non_null_distances = [d for d in distances if d is not None and d < threshold]
        if len(non_null_distances) == 0:
            frame, frequency = getFrameNumberAndFrequency(objects)
            return {
                "cluster": objects[0],
                "num_predictions": 0,
                "time_entropy": sum(
                    [
                        -1
                        * (f / sum(frequency.values()))
                        * np.log(f / sum(frequency.values()))
                        for f in frequency.values()
                    ]
                )
                / len(frequency),
                "time_mode": frame,
            }
        frame, frequency = getFrameNumberAndFrequency(non_null_distance_objects)

        unrelated_field_values = Extractable.getFieldListFromListOfExtractables(
            non_null_distance_objects
        )
        normalized_weights = [
            (1 - (w - min(non_null_distances)) / sum(non_null_distances))
            for w in non_null_distances
        ]
        sum_to_one = [w / sum(normalized_weights) for w in normalized_weights]
        new_x = sum(
            [
                n_w * x
                for x, n_w in zip([o.x for o in non_null_distance_objects], sum_to_one)
            ]
        )
        new_y = sum(
            [
                n_w * y
                for y, n_w in zip([o.y for o in non_null_distance_objects], sum_to_one)
            ]
        )
        return {
            "cluster": cls(
                x=new_x, y=new_y, **{u: v[0] for u, v in unrelated_field_values.items()}
            ),
            "num_predictions": len(non_null_distance_objects),
            "time_entropy": sum(
                [
                    -1
                    * (f / sum(frequency.values()))
                    * np.log(f / sum(frequency.values()))
                    for f in frequency.values()
                ]
            )
            / len(frequency),
            "time_mode": frame,
        }


@dataclass
class RotatedRectangle(Extractable, PolygonClass):
    x: float
    y: float
    width: float
    height: float
    angle: float

    @property
    def points(self):
        return getCornersFromRotatedBox(
            x_p=self.x, y_p=self.y, w_p=self.width, h_p=self.height, a_p=self.angle,
        )

    def contains(self, x, y):
        point = PolygonPoint(x, y)
        polygon = Polygon(self.points)
        return polygon.contains(point)

    @classmethod
    def extract_defining_columns(cls):
        return ["x", "y", "width", "height", "angle"]

    @classmethod
    def clusterLabels(
        cls, objects: List["RotatedRectangle"], distances: List[float], threshold=70
    ) -> Dict[str, Any]:
        assert len(objects) > 0
        assert len(objects) == len(distances)
        non_null_distance_objects = [
            o for o, d in zip(objects, distances) if d is not None and d < threshold
        ]
        non_null_distances = [d for d in distances if d is not None and d < threshold]
        if len(non_null_distances) == 0:
            frame, frequency = getFrameNumberAndFrequency(objects)
            return {
                "cluster": objects[0],
                "num_predictions": 0,
                "time_entropy": sum(
                    [
                        -1
                        * (f / sum(frequency.values()))
                        * np.log(f / sum(frequency.values()))
                        for f in frequency.values()
                    ]
                )
                / len(frequency),
                "time_mode": frame,
            }
        frame, frequency = getFrameNumberAndFrequency(non_null_distance_objects)
        unrelated_field_values = Extractable.getFieldListFromListOfExtractables(
            non_null_distance_objects
        )
        normalized_weights = [
            (1 - (w - min(non_null_distances)) / sum(non_null_distances))
            for w in non_null_distances
        ]
        sum_to_one = [w / sum(normalized_weights) for w in normalized_weights]
        weighted_corners = [
            [[dim * n_w for dim in element] for element in corner]
            for corner, n_w in zip(
                [o.points for o in non_null_distance_objects], sum_to_one
            )
        ]
        summed_corners = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for element in weighted_corners:
            for i in range(len(summed_corners)):
                for j in range(len(summed_corners[0])):
                    summed_corners[i][j] += element[i][j]
        for i in range(len(summed_corners)):
            for j in range(len(summed_corners[0])):
                summed_corners[i][j] = round(summed_corners[i][j])
        x, y, w, h, angle = getRotatedBoxFromCorners(summed_corners)
        return {
            "cluster": cls(
                x=x,
                y=y,
                width=w,
                height=h,
                angle=angle,
                **{u: v[0] for u, v in unrelated_field_values.items()},
            ),
            "num_predictions": len(non_null_distance_objects),
            "time_entropy": sum(
                [
                    -1
                    * (f / sum(frequency.values()))
                    * np.log(f / sum(frequency.values()))
                    for f in frequency.values()
                ]
            )
            / len(frequency),
            "time_mode": frame,
        }


@dataclass
class PointClusters(Extractable, Cluster, PointClass):
    clusters_x: float
    clusters_y: float

    @property
    def points(self):
        return [self.clusters_x, self.clusters_y]

    def interaction(self, object: Extractable):
        if self.subject_id != object.subject_id:
            return None
        if isinstance(object, (lambda: PolygonClass)()):
            min_corner_distance = min(
                [
                    PointClass.distance(
                        x_1=self.clusters_x, y_1=self.clusters_y, x_2=x, y_2=y
                    )
                    for x, y in object.points
                ]
            )
            if object.contains(x=self.clusters_x, y=self.clusters_y):
                return min_corner_distance / 2
            return min_corner_distance
        elif isinstance(object, (lambda: PointClass)()):
            return PointClass.distance(
                x_1=self.clusters_x,
                y_1=self.clusters_y,
                x_2=object.points[0],
                y_2=object.points[1],
            )
        raise NotImplementedError()

    @classmethod
    def extract_defining_columns(cls):
        return ["clusters_x", "clusters_y"]


@dataclass
class PointClustersWithMembers(PointClusters):
    points_x: List[float]
    points_y: List[float]
    cluster_labels: List[int]

    def get_distances(self):
        point_list: List[Point] = [
            Point(
                id=self.id,
                x=x,
                y=y,
                frame=self.frame,
                task=self.task,
                subject_id=self.subject_id,
                tool=self.tool,
                list_order=self.list_order,
            )
            for x, y, label in zip(self.points_x, self.points_y, self.cluster_labels)
            if label == self.list_order
        ]
        return sorted([self.distance_between_classes(point) for point in point_list])


@dataclass
class RotatedRectangleClusters(Extractable, Cluster, PolygonClass):
    clusters_x: float
    clusters_y: float
    clusters_width: float
    clusters_height: float
    clusters_angle: float

    @property
    def points(self):
        return getCornersFromRotatedBox(
            x_p=self.clusters_x,
            y_p=self.clusters_y,
            w_p=self.clusters_width,
            h_p=self.clusters_height,
            a_p=self.clusters_angle,
        )

    def contains(self, x, y):
        point = PolygonPoint(x, y)
        polygon = Polygon(self.points)
        return polygon.contains(point)

    def interaction(self, object: Extractable):
        if self.subject_id != object.subject_id:
            return None
        if isinstance(object, (lambda: PolygonClass)()):
            return PolygonClass.distance(points1=self.points, points2=object.points)
        elif isinstance(object, (lambda: PointClass)()):
            min_corner_distance = min(
                [
                    PointClass.distance(
                        x_1=object.points[0], y_1=object.points[1], x_2=x, y_2=y
                    )
                    for x, y in self.points
                ]
            )
            if self.contains(x=object.points[0], y=object.points[1]):
                return min_corner_distance / 2
            return min_corner_distance
        raise NotImplementedError()

    @classmethod
    def extract_defining_columns(cls):
        return [
            "clusters_x",
            "clusters_y",
            "clusters_width",
            "clusters_height",
            "clusters_angle",
        ]


@dataclass
class ExtractPoint(Point):
    classification_id: int
    user_id: Optional[int] = None

    def interaction(self, object: Extractable):
        if self.subject_id != object.subject_id:
            return None
        if isinstance(object, (lambda: RotatedRectangleClusters)()):
            min_corner_distance = min(
                [
                    PointClass.distance(x_1=self.x, y_1=self.y, x_2=x, y_2=y)
                    for x, y in object.points
                ]
            )
            if object.contains(x=self.x, y=self.y):
                return min_corner_distance / 2
            return min_corner_distance
        elif isinstance(object, (lambda: ExtractRotatedRectangle)()):
            if (
                self.user_id != object.user_id
                or self.classification_id != object.classification_id
            ):
                return None
            if self.task == object.task:
                return 0
            return None
        elif isinstance(object, (lambda: ExtractPoint)()):
            if (
                self.user_id != object.user_id
                or self.classification_id != object.classification_id
            ):
                return None
            if self.task == object.task and self.tool != object.tool:
                return 0
            return None
        elif isinstance(object, (lambda: PointClusters)()):
            return PointClass.distance(
                x_1=self.x, y_1=self.y, x_2=object.clusters_x, y_2=object.clusters_y
            )
        raise NotImplementedError()

    def toCluster(self):
        return PointClusters(
            id=self.id * 1000,
            clusters_x=self.x,
            clusters_y=self.y,
            frame=self.frame,
            task=self.task,
            subject_id=self.subject_id,
            tool=self.tool,
            list_order=self.list_order,
        )


@dataclass
class ExtractRotatedRectangle(RotatedRectangle):
    classification_id: int
    user_id: Optional[int] = None

    def interaction(self, object: Extractable):
        if self.subject_id != object.subject_id:
            return None
        if isinstance(object, (lambda: RotatedRectangleClusters)()):
            return PolygonClass.distance(points1=self.points, points2=object.points)
        elif isinstance(object, (lambda: ExtractRotatedRectangle)()):
            return None
        elif isinstance(object, (lambda: ExtractPoint)()):
            if (
                self.user_id != object.user_id
                or self.classification_id != object.classification_id
            ):
                return None
            if self.task == object.task:
                return 0
            return None
        elif isinstance(object, (lambda: PointClusters)()):
            min_corner_distance = min(
                [
                    PointClass.distance(
                        x_1=object.points[0], y_1=object.points[1], x_2=x, y_2=y
                    )
                    for x, y in self.points
                ]
            )
            if self.contains(x=object.points[0], y=object.points[1]):
                return min_corner_distance / 2
            return min_corner_distance
        raise NotImplementedError()

    def toCluster(self):
        return RotatedRectangleClusters(
            id=self.id * 1000,
            clusters_x=self.x,
            clusters_y=self.y,
            clusters_angle=self.angle,
            clusters_height=self.height,
            clusters_width=self.width,
            frame=self.frame,
            task=self.task,
            subject_id=self.subject_id,
            tool=self.tool,
            list_order=self.list_order,
        )
