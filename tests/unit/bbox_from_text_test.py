import numpy as np

from labelme._automation.bbox_from_text import get_shapes_from_bboxes


def test_get_shapes_from_bboxes_adds_prediction_metadata() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    scores = np.array([0.87], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)
    texts = ["face"]

    shapes = get_shapes_from_bboxes(
        boxes=boxes,
        scores=scores,
        labels=labels,
        texts=texts,
        masks=None,
        shape_type="rectangle",
    )

    assert len(shapes) == 1
    shape = shapes[0]

    assert shape.label == "face"
    assert shape.shape_type == "rectangle"
    assert len(shape.points) == 2
    assert shape.points[0].x() == 10.0
    assert shape.points[0].y() == 20.0
    assert shape.points[1].x() == 30.0
    assert shape.points[1].y() == 40.0

    assert shape.other_data["score"] == 0.87
    assert shape.other_data["confidence"] == 0.87
    assert shape.other_data["class_name"] == "face"
    assert shape.other_data["class_id"] == 0
