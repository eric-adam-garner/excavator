def clean_polyline(points, tol):
    # remove duplicate final vertex
    if len(points) > 1:
        dx = points[0][0] - points[-1][0]
        dy = points[0][1] - points[-1][1]
        if dx * dx + dy * dy < tol * tol:
            points = points[:-1]

    # remove consecutive duplicates
    cleaned = [points[0]]
    for p in points[1:]:
        dx = p[0] - cleaned[-1][0]
        dy = p[1] - cleaned[-1][1]
        if dx * dx + dy * dy > tol * tol:
            cleaned.append(p)

    return cleaned
