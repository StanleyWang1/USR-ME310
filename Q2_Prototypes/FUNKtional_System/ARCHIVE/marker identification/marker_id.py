import numpy as np

def assign_points(points):

    D = np.zeros((3, 3))

    for i in range(3):
        for j in range(i + 1, 3):
            D[i, j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))

    upper_triangular_values = D[np.triu_indices(3, k=1)]
    x = np.max(upper_triangular_values)
    y = np.min(upper_triangular_values)

    D12, D13, D23 = D[0, 1], D[0, 2], D[1, 2]  # Using 0-based indexing

    if x == D12 and y == D13:
        B, A, C = points[0], points[1], points[2]
    elif x == D12 and y == D23:
        A, B, C = points[0], points[1], points[2]
    elif x == D13 and y == D12:
        B, C, A = points[0], points[1], points[2]
    elif x == D13 and y == D23:
        A, C, B = points[0], points[1], points[2]
    elif x == D23 and y == D12:
        C, B, A = points[0], points[1], points[2]
    elif x == D23 and y == D13:
        C, A, B = points[0], points[1], points[2]
    else:
        raise ValueError("Invalid x and y values for the given distance matrix.")

    return A, B, C


if __name__ == "__main__":

    points = [(0, 0, 0), (0, 4, 0), (1, 1, 0)]

    A, B, C = assign_points(points)

    print(f"A: {A}, B: {B}, C: {C}")