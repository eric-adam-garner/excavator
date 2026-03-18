from pslg import PSLG


def test_annulus_nesting():
    p = PSLG()
    p.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    p.add_polygon([(1, 1), (1, 3), (3, 3), (3, 1)])

    report = p.classify_loops()

    assert len(report.outer_loops) == 1
    assert len(report.hole_loops) == 1
    assert sorted(report.depths) == [0, 1]


def test_two_disjoint_outer_loops():
    p = PSLG()
    p.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p.add_polygon([(5, 0), (7, 0), (7, 2), (5, 2)])

    report = p.classify_loops()

    assert len(report.outer_loops) == 2
    assert len(report.hole_loops) == 0
    assert report.depths == [0, 0] or sorted(report.depths) == [0, 0]


def test_nested_island_structure():
    p = PSLG()
    p.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p.add_polygon([(2, 2), (2, 8), (8, 8), (8, 2)])
    p.add_polygon([(4, 4), (6, 4), (6, 6), (4, 6)])

    report = p.classify_loops()

    assert sorted(report.depths) == [0, 1, 2]
    assert len(report.outer_loops) == 2
    assert len(report.hole_loops) == 1