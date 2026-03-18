from pslg import PSLG


def test_vertex_on_segment():
    p = PSLG()

    a = p.add_vertex(0, 0)
    b = p.add_vertex(4, 0)
    c = p.add_vertex(2, 0)
    d = p.add_vertex(2, 3)

    p.add_segment(a, b)
    p.add_segment(c, d)

    issues = p.find_vertices_on_segments()

    assert len(issues) == 1
    assert issues[0]["vertex"] == c


def test_no_vertex_on_segment():
    p = PSLG()

    a = p.add_vertex(0, 0)
    b = p.add_vertex(4, 0)
    c = p.add_vertex(2, 1)

    p.add_segment(a, b)

    assert p.find_vertices_on_segments() == []
