def get_left_most_point(points: list):
    left_most_index: int = 0

    for index in range(len(points)):
        point = points[index]
        lmp = points[left_most_index]

        if point[0] < lmp[0] or (point[0] == lmp[0] and point[1] < lmp[1]):
            left_most_index = index

    return left_most_index

def orientation(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

def calculate_convex_hull(points: list, add_first_twice: bool = True):
    # There must be at least 3 points
    if not points or len(points) < 3:
        return

    hull = []

    n: int = len(points)

    # Find the leftmost point
    left_most_index: int = get_left_most_point(points)

    meet_left_most = False

    p = left_most_index

    while not meet_left_most:
        # Add current point to result 
        hull.append(points[p])

        q = (p + 1) % n

        for i in range(n):
            if orientation(points[p], points[i], points[q]) < 0:
                q = i

        p = q

        # While we don't come to first point
        meet_left_most = p == left_most_index

    if add_first_twice:
        hull.append(points[p])

    return hull