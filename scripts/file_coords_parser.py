def parse_file_coords(coord_string: str) -> list[tuple[int, int]]:
    parts = coord_string.split(" ", 1)[1]
    coords = parts.split(") (")

    result = []
    for coord in coords[:2]:
        coord = coord.strip("()")
        x, y = map(int, coord.split())
        result.append((x, y))

    return result


def parse_etiquette(etiquette: str) -> int:
    return int(etiquette.split(" ")[0])
