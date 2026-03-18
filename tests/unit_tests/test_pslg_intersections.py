import pytest

from pslg import PSLG


def test_no_intersection_diagonal_inside_square():
    pslg = PSLG()

    pslg.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    pslg.add_segment(pslg.add_vertex(0, 0), pslg.add_vertex(4, 4))

    issues = pslg.find_segment_intersections()

    assert issues == []


def test_proper_crossing():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 4)
    c = pslg.add_vertex(0, 4)
    d = pslg.add_vertex(4, 0)

    pslg.add_segment(a, b)
    pslg.add_segment(c, d)

    issues = pslg.find_segment_intersections()

    assert len(issues) == 1
    assert issues[0]["type"] == "proper"


def test_crossing_through_polygon():
    pslg = PSLG()

    pslg.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

    v0 = pslg.add_vertex(-1, 2)
    v1 = pslg.add_vertex(5, 2)

    pslg.add_segment(v0, v1)

    issues = pslg.find_segment_intersections()

    assert len(issues) == 2
    assert all(i["type"] == "proper" for i in issues)


def test_shared_endpoint_allowed():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)
    c = pslg.add_vertex(4, 4)

    pslg.add_segment(a, b)
    pslg.add_segment(b, c)

    issues = pslg.find_segment_intersections()

    assert issues == []


def test_t_junction_detection():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)

    pslg.add_segment(a, b)

    c = pslg.add_vertex(2, 0)
    d = pslg.add_vertex(2, 3)

    pslg.add_segment(c, d)

    issues = pslg.find_segment_intersections()

    assert len(issues) == 1
    assert issues[0]["type"] == "t_junction"


def test_colinear_overlap():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)
    c = pslg.add_vertex(2, 0)
    d = pslg.add_vertex(6, 0)

    pslg.add_segment(a, b)
    pslg.add_segment(c, d)

    issues = pslg.find_segment_intersections()

    assert len(issues) == 1
    assert issues[0]["type"] == "overlap"
