from pslg import PSLG


def test_valid_annulus():
    p = PSLG()
    p.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    p.add_polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    report = p.validate()

    assert report.is_valid
    assert report.stats["num_loops"] == 2


def test_invalid_crossing():
    p = PSLG()

    a = p.add_vertex(0, 0)
    b = p.add_vertex(4, 4)
    c = p.add_vertex(0, 4)
    d = p.add_vertex(4, 0)

    p.add_segment(a, b)
    p.add_segment(c, d)

    report = p.validate()

    assert not report.is_valid


def test_invalid_open_chain():
    p = PSLG()

    a = p.add_vertex(0, 0)
    b = p.add_vertex(1, 0)
    c = p.add_vertex(2, 0)

    p.add_segment(a, b)
    p.add_segment(b, c)

    report = p.validate()

    assert not report.is_valid
    assert report.stats["num_open_chains"] == 1