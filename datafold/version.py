class Version:
    """Current version."""

    major_version: int = 0  # making incompatible API changes,
    minor_version: int = 1  # adding functionality in a backwards-compatible manner
    patch: int = 0  # for backwards-compatible bug fixes

    # NOTE: for the moment, there are no release cycles (such as alpha, beta, release
    #  candidates,...)

    # Semantic versioning policy
    # preferred by Python
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#semantic-versioning-preferred

    # See also https://semver.org/

    v_short = f"{major_version}.{minor_version}.{patch}"

    # make date of release for longer version numbers
    year: int = 2020
    month: int = 2
    day: int = 19
    v_nr: int = 1

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
