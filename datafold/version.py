import datetime


class Version:
    """Current datafold version."""

    # Semantic versioning policy
    # preferred by Python
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#semantic-versioning-preferred

    # See also https://semver.org/

    ###### TO UPDATE -- START
    major_version: int = 1  # making incompatible API changes,
    minor_version: int = 0  # adding functionality in a backwards-compatible manner
    patch: int = 1  # for backwards-compatible bug fixes

    # Make date of release for longer version numbers.
    year: int = 2020
    month: int = 6
    day: int = 29
    ###### TO UPDATE -- END

    assert major_version >= 0 and isinstance(major_version, int)
    assert minor_version >= 0 and isinstance(minor_version, int)
    assert patch >= 0 and isinstance(patch, int)

    v_short = f"{major_version}.{minor_version}.{patch}"

    date_string = datetime.datetime(year=year, month=month, day=day).strftime(
        "%Y-%m-%d"
    )

    v_long = f"{v_short} ({date_string})"
    v_gnu = f"datafold {v_long}"

    @staticmethod
    def print_version():
        print(f"v_short = {Version.v_short}")
        print(f"v_long  = {Version.v_long}")
        print(f"v_gnu   = {Version.v_gnu}")


if __name__ == "__main__":
    Version.print_version()
