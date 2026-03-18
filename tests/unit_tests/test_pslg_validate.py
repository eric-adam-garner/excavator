from pslg import PSLG


def test_validate_annulus_pslg():
    pslg = PSLG(tol=1e-8)

    pslg.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    pslg.add_polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    report = pslg.validate()

    assert report.is_valid
    assert report.stats["num_proper_intersections"] == 0
    assert report.stats["num_t_junctions"] == 0
    assert report.stats["num_overlaps"] == 0
    assert report.stats["num_vertices_on_segments"] == 0


def test_validate_proper_crossing():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 4)
    c = pslg.add_vertex(0, 4)
    d = pslg.add_vertex(4, 0)

    pslg.add_segment(a, b)
    pslg.add_segment(c, d)

    report = pslg.validate()

    assert not report.is_valid
    assert report.stats["num_proper_intersections"] == 1


def test_validate_t_junction():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)
    c = pslg.add_vertex(2, 0)
    d = pslg.add_vertex(2, 3)

    pslg.add_segment(a, b)
    pslg.add_segment(c, d)

    report = pslg.validate()

    assert not report.is_valid
    assert report.stats["num_t_junctions"] == 1
    assert report.stats["num_vertices_on_segments"] == 1


def test_validate_overlap():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)
    c = pslg.add_vertex(2, 0)
    d = pslg.add_vertex(6, 0)

    pslg.add_segment(a, b)
    pslg.add_segment(c, d)

    report = pslg.validate()

    assert not report.is_valid
    assert report.stats["num_overlaps"] == 1


def test_validate_duplicate_segment():
    pslg = PSLG()

    a = pslg.add_vertex(0, 0)
    b = pslg.add_vertex(4, 0)

    pslg.add_segment(a, b)
    pslg.add_segment(a, b)

    report = pslg.validate()

    assert not report.is_valid
    assert report.stats["num_duplicate_segments"] == 1
