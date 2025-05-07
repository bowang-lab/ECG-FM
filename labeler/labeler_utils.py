from typing import Callable, Dict, List, Optional, Union
from collections import OrderedDict
from functools import reduce

import re

import numpy as np
import pandas as pd

def remove_after_substring(series: pd.Series, substring: str, case: bool = False):
    if case:
        slc = series.str.find(substring)
    else:
        slc = series.str.lower().str.find(substring.lower())

    not_found_inds = slc == -1
    slc.iloc[not_found_inds] = series.str.len()[not_found_inds]

    data = pd.DataFrame(series, columns=["texts"])
    data["slice"] = slc

    return data.apply(lambda row: row["texts"][:row["slice"]], axis=1)

def remove_after_last_substring(series: pd.Series, substring: str, case: bool = False):
    cseries = series.copy()
    if case is False:
        cseries = cseries.str.lower()
        substring = substring.lower()

    cseries = cseries.apply(lambda x: x[::-1])
    substring = substring[::-1]

    ind = cseries.str.find(substring)
    ind[ind == -1] = np.nan

    def _slice(x):
        nonlocal substring

        if pd.isna(x["ind"]):
            return x["series"]

        return x["series"][:-int(x["ind"]) - len(substring)]

    data = pd.concat([series.rename("series"), ind.rename("ind")], axis=1)

    return data.apply(_slice, axis=1)

def dict_ordered_by_len(dictionary: dict, reverse: bool = True):
    if reverse:
        return OrderedDict(sorted(dictionary.items(), key=lambda item: -len(item[0])))
    else:
        return OrderedDict(sorted(dictionary.items(), key=lambda item: len(item[0])))

def pattern_printing(
    texts: pd.Series,
    equals: Optional[str] = None,
    includes: Optional[Union[str, List[str]]] = None,
    excludes: Optional[Union[str, List[str]]] = None,
    condition_on: Optional[pd.Series] = None,
    limit: int = 50,
):
    includes = includes or []
    excludes = excludes or []

    includes = includes if isinstance(includes, list) else [includes]
    excludes = excludes if isinstance(excludes, list) else [excludes]

    if condition_on is None:
        condition_on = texts
    else:
        assert len(texts) == len(condition_on)

    conditions = []
    if equals is not None:
        conditions.append(condition_on == equals)

    for txt in includes:
        conditions.append(condition_on.str.contains(txt, regex=False, case=False))

    for txt in excludes:
        conditions.append(~condition_on.str.contains(txt, regex=False, case=False))

    condition = reduce(lambda x, y: x & y, conditions)

    res = texts[condition]
    print(f"Length: {len(res)}")
    for val in res.iloc[:limit]:
        print(val)
