from pslg import PSLG


def test_single_loop():
    p = PSLG()
    p.add_polygon([(0, 0), (1, 0), (0, 1)])

    report = p.extract_loops()

    assert len(report.loops) == 1
    assert len(report.open_chains) == 0
    assert report.loop_areas[0] > 0


def test_two_loops_annulus():
    p = PSLG()
    p.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    p.add_polygon([(1, 1), (1, 3), (3, 3), (3, 1)])

    report = p.extract_loops()

    assert len(report.loops) == 2
    assert len(report.open_chains) == 0


def test_open_chain():
    p = PSLG()

    a = p.add_vertex(0, 0)
    b = p.add_vertex(1, 0)
    c = p.add_vertex(2, 0)

    p.add_segment(a, b)
    p.add_segment(b, c)

    report = p.extract_loops()

    assert len(report.loops) == 0
    assert len(report.open_chains) == 1