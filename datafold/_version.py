import datetime


class Version:
    """Current datafold version."""

    # Semantic versioning policy
    # preferred by Python
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#semantic-versioning-preferred

    # See also https://semver.org/

    # TO UPDATE -- START
    major_version: int = 1  # making incompatible API changes,
    minor_version: int = 1  # adding functionality in a backwards-compatible manner
    patch: int = 5  # for backwards-compatible bug fixes

    # additional release suffixes
    post: int = 0  # for minor corrections
    rc: int = 0  # for release candidate (e.g. for testing upload to PyPI)

    # Set date of release for longer version numbers.
    year: int = 2021
    month: int = 7
    day: int = 5
    # TO UPDATE -- END

    assert major_version >= 0 and isinstance(major_version, int)
    assert minor_version >= 0 and isinstance(minor_version, int)
    assert patch >= 0 and isinstance(patch, int)
    assert post >= 0 and isinstance(post, int)
    assert rc >= 0 and isinstance(rc, int)

    attach_post = f".post{post}" if post > 0 else ""
    attach_rc = f"rc{rc}" if rc > 0 else ""

    v_short: str = f"{major_version}.{minor_version}.{patch}{attach_post}{attach_rc}"

    date_string = datetime.datetime(year=year, month=month, day=day).strftime(
        "%Y-%m-%d"
    )

    v_long: str = f"{v_short} ({date_string})"
    v_gnu: str = f"datafold {v_long}"

    @staticmethod
    def print_version():
        print(f"v_short = {Version.v_short}")
        print(f"v_long  = {Version.v_long}")
        print(f"v_gnu   = {Version.v_gnu}")


if __name__ == "__main__":
    Version.print_version()
