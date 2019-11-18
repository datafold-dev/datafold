class Version:
    """Current version."""

    # set only integers:
    year = 2019
    month = 10
    day = 18
    v_nr = 1

    assert 1 <= day <= 31
    assert v_nr >= 1

    months_english = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    v_short = f"{year}{str(month).zfill(2)}.{v_nr}"  # YYYYMM.V
    date_string = f"{year}-{months_english[month-1]}-{day}"

    v_long = f"{v_short} ({date_string})"
    v_gnu = f"datafold {v_long}"

    @staticmethod
    def print_version():
        print(f"v_short = {Version.v_short}")
        print(f"v_long  = {Version.v_long}")
        print(f"v_gnu   = {Version.v_gnu}")


if __name__ == "__main__":
    Version.print_version()
