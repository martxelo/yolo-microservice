import sys
import os

import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import decode



@pytest.mark.parametrize(
    'xmin, xmax, ymin, ymax, expected_area',
    [
        (0, 1, 0, 1, 1),
        (0, 2, 0, 2, 4),
        (5, 10, 10, 15, 25),
        (0, 1, 2, 5, 3),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0),
        (0, -1, 0, 1, 0)
    ]
)
def test_calc_area(xmin, xmax, ymin, ymax, expected_area):

    area = decode.calc_area(xmin, xmax, ymin, ymax)

    assert np.isclose(area, expected_area)


@pytest.mark.parametrize(
    'box1, box2, expected_overlap',
    [
        (
            ['label', 0.9, 0, 10, 0, 10],
            ['label', 0.9, 5, 15, 5, 15],
            25.0/175.0
        ),
        (
            ['label', 0.9, 0, 10, 0, 10],
            ['label', 0.9, 15, 25, 15, 25],
            0.0
        ),
        (
            ['label', 0.9, 0, 10, 0, 10],
            ['label', 0.9, 0, 5, 0, 10],
            0.5
        ),
        (
            ['label', 0.9, 25, 30, 20, 25],
            ['label', 0.9, 25, 30, 20, 25],
            1.0
        ),
        (
            ['label', 0.9, 0, 10, 0, 10],
            ['label', 0.9, 10, 20, 10, 20],
            0.0
        )
    ]
)
def test_calc_overlap(box1, box2, expected_overlap):

    overlap = decode.calc_overlap(box1, box2)

    assert np.isclose(overlap, expected_overlap)


@pytest.mark.parametrize(
    'boxes, max_overlap, expected_boxes',
    [
        (
            [['label', 1.0, 0, 10, 0, 10],
             ['label', 0.9, 0, 9, 0, 9]],
            0.25,
            [['label', 1.0, 0, 10, 0, 10]]
        ),
        (
            [['label', 0.9, 0, 9, 0, 9],
             ['label', 1.0, 0, 10, 0, 10]],
            0.25,
            [['label', 1.0, 0, 10, 0, 10]]
        ),
        (
            [['label', 0.9, 10, 20, 15, 25],
             ['label', 1.0, 0, 10, 0, 10]],
            0.25,
            [['label', 1.0, 0, 10, 0, 10],
             ['label', 0.9, 10, 20, 15, 25]]
        )
    ]
)
def test_non_max_supression(boxes, max_overlap, expected_boxes):

    nms_boxes = decode.non_max_supression(boxes, max_overlap)

    assert nms_boxes == expected_boxes


@pytest.mark.parametrize(
    'boxes, size, img_size, expected_boxes',
    [
        (
            [['label', 1.0, 100, 200, 50, 150],
             ['label', 0.9, 0, 500, 15, 250]],
            500,
            (1500, 1000),
            [['label', 1.0, 100*3, 200*3, 50*2, 150*2],
             ['label', 0.9, 0*3, 500*3, 15*2, 250*2]]
        )
    ]
)
def test_scale_boxes(boxes, size, img_size, expected_boxes):

    scaled_boxes = decode.scale_boxes(boxes, size, img_size)

    assert scaled_boxes == expected_boxes