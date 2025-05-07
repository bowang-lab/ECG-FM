from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import re
import pandas as pd

from common import to_list

def unused_char(string: str, printable: bool = True) -> str:
    """
    Find and return an unused character in the given string.

    Parameters
    ----------
    string : str
        The input string.
    printable : bool, default True
        If True, consider only printable characters.

    Returns
    -------
    str
        An unused character in the given string.

    Raises
    ------
    ValueError
        If no unused character is found.
    """
    char_set = set(string)

    if printable:
        for idx in range(0x110000):
            char = chr(idx)
            if char.isprintable() and char not in char_set:
                return char
    else:
        for idx in range(0x110000):
            char = chr(idx)
            if char not in char_set:
                return char

    raise ValueError("No unused characters.")


def unused_chars(char_set: set, printable: bool = True) -> set:
    """
    Find and return the set of unused characters in comparison to a given character set.

    Parameters
    ----------
    char_set : set
        The set of characters to compare against.
    printable : bool, default True
        If True, consider only printable characters.

    Returns
    -------
    set
        A set of unused characters in comparison to the given character set.
    """
    if printable:
        all_chars = set([chr(i) for i in range(0x110000) if chr(i).isprintable()])
    else:
        all_chars = set([chr(i) for i in range(0x110000)])

    return all_chars - char_set

def border_alphanumeric(pattern: Union[str, pd.Series], escape: bool = True):
    """
    Create a border for non-word chars when beginning/ending with alphanumeric chars.

    Uses negative lookbehind and lookahead assertions on non-word characters such as
    spaces, punctuation, and the beginning/end of a line. Only applies bordering
    when the string begins/ends with alphanumeric characters.

    This is useful during pattern matching such that, for example, "my" wouldn't be
    matched when a part of "myself" because "my" is followed by an alpha character.
    As for non-word start/end characters, such as on ", test", where the comma is
    likely preceeded by an alpha character, this would be allowed.

    Parameters
    ----------
    pattern: str or pandas.Series
        String or series of strings representing patterns to be matched.
    escape : bool
        If True, escape the patterns to avoid issues with regex special characters.
        If False, assumes that `pattern` was already escaped.

    Returns
    -------
    str or pandas.Series
        Bordered pattern(s).
    """
    if isinstance(pattern, str):
        pattern = pd.Series([pattern])

    if escape:
        escaped = pattern.apply(re.escape)
    else:
        escaped = pattern

    alpha_first = pattern.str.slice(stop=1).str.isalnum()
    alpha_last = pattern.str.slice(start=-1).str.isalnum()
    prepend = alpha_first.map({False: r"", True: r"(?<![^\W])"})
    append = alpha_last.map({False: r"", True: r"(?![^\W])"})

    bordered = prepend.str.cat([escaped, append])

    if len(bordered) == 1:
        return bordered.values[0]

    return bordered

def replace(
    texts: pd.Series,
    replace_data: Union[Dict[str, str], pd.DataFrame],
    border_alnum: bool = True,
    sort_by_length: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Replace patterns in a pandas Series of strings with corresponding replacements.

    Parameters
    ----------
    texts : pandas.Series
        A pandas Series containing strings to be processed.
    replace_data : Union[Dict[str, str], pandas.DataFrame]
        A dictionary or DataFrame containing (unescaped) patterns and their
        replacements. If a DataFrame, expects "pat" and "repl" columns.
    border_alnum : bool, default True
        Adds word boundaries to patterns.
    sort_by_length : bool, default True
        Sort by pattern length to match longer patterns before shorter ones.
    Returns
    -------
    tuple of pandas.Series, pandas.DataFrame
        A tuple containing the processed texts and the DataFrame used for replacements.
    """
    if isinstance(replace_data, dict):
        replace_data = pd.DataFrame(
            replace_data.items(),
            columns=["pat", "repl"],
        )

    # Sort by pattern length to match longer patterns over shorter ones
    if sort_by_length:
        replace_data = replace_data.reindex(
            replace_data["pat"].str.len().sort_values(ascending=False).index
        )

    # Optionally perform bordering, or simply escape the string
    if border_alnum:
        replace_data["pat"] = border_alphanumeric(replace_data["pat"], escape=True)
    else:
        replace_data["pat"] = replace_data["pat"].apply(re.escape)

    # Perform replacements
    for _, row in replace_data.iterrows():
        texts = texts.str.replace(row["pat"], row["repl"], regex=True)

    return texts, replace_data

def fix_bordered_typos(texts: pd.Series, typos: Dict[str, str], escape: bool = True):
    """
    Fix bordered typos in a Pandas Series.

    This function checks whether using regular expressions with word boundaries
    is faster than doing individual replacements using `.str.replace`.

    Parameters
    ----------
    texts : pandas.Series
        A string series to fix.
    typos : Dict[str, str]
        A dictionary mapping the typos to their correct versions.
    escape : bool
        Escape the typos to avoid issues with regex special characters.

    Returns
    -------
    pd.Series
        The series with fixed typos.

    Notes
    -----
    This function uses regular expressions to match and replace the typos in the
    given Pandas Series.
    """
    keys = typos.keys()
    if escape:
        keys = map(re.escape, keys)

    pattern = re.compile(r"\b(" + "|".join(keys) + r")\b")

    return texts.apply(lambda x: pattern.sub(lambda y: typos[y.group()], str(x)))

def remove_repeated_whitespace(texts: pd.Series, strip: bool = True) -> pd.Series:
    """
    Replace repeated whitespace characters with a single space.

    If `strip` is True, additionally strip to remove leading/trailing whitespace.

    Parameters
    ----------
    texts : pd.Series
        Series containing the strings to process.
    strip : bool, default True
        Whether to strip whitespace.

    Returns
    -------
    pd.Series
        Series with repeated whitespace removed; optionally stripped.
    """
    texts = texts.str.replace(r'\s+', ' ', regex=True)
    if strip:
        texts = texts.str.strip()

    return texts
