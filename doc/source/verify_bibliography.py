import pathlib

import pybtex.database as bib

path_script = pathlib.Path(__file__).parent.absolute()
path_bibfile = pathlib.PurePath(path_script, "_static/literature.bib")


def has_field(entry, field):
    return field in entry.fields


def remove_field(entry, fieldname: str):
    try:
        del entry.fields[fieldname]
    except KeyError:
        pass  # the field does not exist, then there is nothing to do
    return entry


def sort_bibfile(bibfile):
    sorted_entries = sorted(bibfile.entries)
    new_bibfile = bib.BibliographyData()

    for entry_name in sorted_entries:
        new_bibfile.add_entry(entry_name, entry=bibfile.entries[entry_name])

    return new_bibfile


def _validate_year(entry):
    try:
        year = entry.fields["year"]
    except KeyError:
        # re-raise with more information
        raise KeyError(f"key = {entry.key} has no mandatory field 'year'")

    try:
        int(year)
    except ValueError:
        # re-raise with more information
        raise ValueError(
            f"key = {entry.key} has invalid value in 'year' " f"(got: year={year})"
        )


def _validate_key(key, entry):
    key_error_text = (
        f"key='{key}' is invalid. A valid key has the form "
        "'[name (lowercase)]-[year (4 digits)][empty or a-z]' (e.g. "
        "'einstein-1904' with a second publication 'einstein-1904a') with "
        "the last name of the first author, year of publication and a "
        "single character for key name mangling if necessary."
    )

    if "-" not in key:
        raise KeyError(key_error_text)

    key_author_last, year = key.split("-")

    if key_author_last != key_author_last.lower():
        raise KeyError(key_error_text)

    first_author_last = entry.persons["author"][0].last_names
    first_author_last = "".join(first_author_last)
    first_author_last = first_author_last.replace("{\\'", "").replace("}", "")

    if key_author_last != first_author_last.lower():
        raise KeyError(key_error_text)

    if len(year) == 4:
        try:
            int(year)
        except ValueError:
            raise KeyError(key_error_text)
    elif len(year) == 5:
        if year[4] not in "abcdefghijklmnopqrstuvwxyz":
            raise KeyError(key_error_text)
    else:
        raise KeyError(key_error_text)


def adapt_bib():
    bibfile = bib.parse_file(path_bibfile)

    for key in bibfile.entries:
        # note that there is no copy, so all following operations are directly in the entry
        entry = bibfile.entries[key]

        # do not include these common fields as they are not
        # printed anyway or too specific:
        remove_field(entry, "abstract")
        remove_field(entry, "urldate")
        remove_field(entry, "file")
        remove_field(entry, "shorttitle")
        remove_field(entry, "annote")
        remove_field(entry, "note")
        remove_field(entry, "language")
        remove_field(entry, "keywords")
        remove_field(entry, "address")

        # only use DOI as a single link to the paper
        # remove all the other fields that identify a bib-entry
        if has_field(entry, "doi"):
            remove_field(entry, "issn")
            remove_field(entry, "url")
            remove_field(entry, "isbn")

    bibfile = sort_bibfile(bibfile)
    bibfile.to_file(path_bibfile)


def _has_identifier(entry):
    has_doi = has_field(entry, "doi")

    if has_doi:
        return True
    else:
        return (
            has_field(entry, "url")
            or has_field(entry, "issn")
            or has_field(entry, "isbn")
        )


def validate_bib():
    bibfile = bib.parse_file(path_bibfile)

    for key in bibfile.entries:
        entry = bibfile.entries[key]

        _validate_year(entry)
        _validate_key(key, entry)

        if not _has_identifier(entry):
            print(f"WARNING: key={key} has no identifier link (URL, ISBN, ISSN or DOI)")


if __name__ == "__main__":
    adapt_bib()
    validate_bib()
