from lunar_lander.main import MapNDto1d


def test_size():
    dimension_list = [3, 3, 3]
    expected = 3 * 3 * 3
    to_1d = MapNDto1d(cardinality_per_dimension=dimension_list)
    assert to_1d.size() == expected


def test_index_1d():
    dimension_list = [3, 3, 3]
    indices = []
    to_1d = MapNDto1d(cardinality_per_dimension=dimension_list)
    expected = [i for i in range(to_1d.size())]
    for x in range(3):
        for y in range(3):
            for z in range(3):
                point = [x, y, z]
                indices.append(to_1d.index_1d(point))
    indices = list(set(indices))
    indices.sort()
    assert indices == expected
