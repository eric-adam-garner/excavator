from pslg import PSLG


def test_orientation_normalization_annulus():
    p = PSLG()

    # both CCW initially
    p.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    p.add_polygon([(1, 1), (1, 3), (3, 3), (3, 1)])

    p.normalize_orientation()

    nest = p.classify_loops()

    areas = nest.loop_areas
    depths = nest.depths

    for area, depth in zip(areas, depths):
        if depth % 2 == 0:
            assert area > 0
        else:
            assert area < 0
