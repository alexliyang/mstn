import math
import numpy as np

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


def clockwise(gt_text):
    # for index, d1 in enumerate(gt_text):
        r = np.full((4, 2), 0.0, dtype='float32')
        for j in range(4):
            r[j][0] = gt_text[j * 2]
            r[j][1] = gt_text[j * 2 + 1]

        xSorted = r[np.argsort(r[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        vector_0 = np.array(bl - tl)
        vector_1 = np.array(rightMost[0] - tl)
        vector_2 = np.array(rightMost[1] - tl)

        angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
        (br, tr) = rightMost[np.argsort(angle), :]

        return [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]

        # f.write(','.join(list(map(str, (tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1])))) + '\n')
