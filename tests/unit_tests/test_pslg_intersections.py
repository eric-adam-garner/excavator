from pslg import PSLG


def test_proper_crossing():
    p = PSLG()

    a = p.add_vertex(0, 0)
    b = p.add_vertex(4, 4)
    c = p.add_vertex(0, 4)
    d = p.add_vertex(4, 0)

    p.add_segment(a, b)
    p.add_segment(c, d)

    issues = p.find_segment_intersections()

    assert len(issues) == 1
    assert issues[0]["type"] == "proper"


def test_colinear_overlap():
    p = PSLG()

    a = p.add_vertex(0, 0)
    b = p.add_vertex(4, 0)
    c = p.add_vertex(2, 0)
    d = p.add_vertex(6, 0)

    p.add_segment(a, b)
    p.add_segment(c, d)

    issues = p.find_segment_intersections()

    assert len(issues) == 1
    assert issues[0]["type"] == "overlap"


def test_no_intersection_diagonal_inside_square():
    p = PSLG()

    p.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    p.add_segment(p.add_vertex(0, 0), p.add_vertex(4, 4))

    issues = p.find_segment_intersections()

    assert issues == []