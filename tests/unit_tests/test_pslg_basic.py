from pslg import PSLG


def test_vertex_welding():
    p = PSLG(tol=1e-6)

    v0 = p.add_vertex(0, 0)
    v1 = p.add_vertex(1e-7, 0)

    assert v0 == v1
    assert len(p.vertices) == 1


def test_add_polygon_creates_segments():
    p = PSLG()
    p.add_polygon([(0, 0), (1, 0), (1, 1)])

    assert len(p.vertices) == 3
    assert len(p.segments) == 3