from pslg import PSLG


def test_vertex_on_segment_detected():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)
    c = pslg.add_vertex(2, 0)
    d = pslg.add_vertex(2, 3)

    pslg.add_segment(a, b)
    pslg.add_segment(c, d)

    issues = pslg.find_vertices_on_segments()

    assert len(issues) == 1
    assert issues[0]["type"] == "vertex_on_segment"
    assert issues[0]["vertex"] == c
    assert issues[0]["segment"] == 0


def test_polygon_vertices_not_reported_on_own_segments():
    pslg = PSLG()

    pslg.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

    issues = pslg.find_vertices_on_segments()

    assert issues == []


def test_interior_point_not_on_segment_not_reported():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)
    c = pslg.add_vertex(2, 1)

    pslg.add_segment(a, b)

    issues = pslg.find_vertices_on_segments()

    assert issues == []