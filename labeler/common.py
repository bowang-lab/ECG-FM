from typing import Any, Dict, Iterable, List, Optional, Set, Union

from itertools import chain

import numpy as np

def to_list(obj: Any) -> List[Any]:
    """Convert some object to a list of object(s) unless already one.

    Parameters
    ----------
    obj : any
        The object to convert to a list.

    Returns
    -------
    list
        The processed object.

    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, (np.ndarray, set, dict)):
        return list(obj)

    return [obj]

def to_list_optional(
    obj: Optional[Any], none_to_empty: bool = False
) -> Union[List[Any], None]:
    """Convert some object to a list of object(s) unless already None or a list.

    Parameters
    ----------
    obj : any
        The object to convert to a list.
    none_to_empty: bool, default = False
        If true, return a None obj as an empty list. Otherwise, return as None.

    Returns
    -------
    list or None
        The processed object.

    """
    if obj is None:
        if none_to_empty:
            return []
        return None

    return to_list(obj)

def flatten_2d_list(lst: List[List]) -> List:
    """
    Flattens a 2D list into a 1D list.

    Parameters
    ----------
    lst : List[List]
        The input 2D list.

    Returns
    -------
    List
        The flattened 1D list.
    """
    return list(chain.from_iterable(lst))

def include_exclude_filter(
    values: set,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
) -> Optional[set]:
    """
    Filter a set of values based on inclusion or exclusion criteria.

    Only one of `include` and `exclude` can be defined. If neither are, then
    None is returned.

    Parameters
    ----------
    values : set
        The set of values to filter.

    include : str or list of str, optional, default: None
        Values to include in the filtered set.

    exclude : str or list of str, optional, default: None
        Values to exclude from the filtered set.

    Returns
    -------
    set, optional
        The filtered set of values, or None if include and exclude aren't specified.

    Raises
    ------
    ValueError
        If both `include` and `exclude` are specified.
        If `include` is specified and not a subset of `values`.
        If `exclude` is specified and not a subset of `values`.
    """
    include = to_list_optional(include)
    exclude = to_list_optional(exclude)

    if include is not None and exclude is not None:
        raise ValueError("Cannot specify both include and exclude.")

    if include is None and exclude is None:
        return None

    elif include is not None:
        selected = set(include)
        if not selected.issubset(values):
            raise ValueError(
                f"include must be a subset of the keys: {', '.join(values)}."
            )
    elif exclude is not None:
        if not set(exclude).issubset(values):
            raise ValueError(
                f"exclude must be a subset of the keys: {', '.join(values)}."
            )
        selected = values - set(exclude)

    return selected

def is_callable(obj: Any, raise_err: bool = False) -> bool:
    """
    Check whether a given object is callable.

    Parameters
    ----------
    obj : Any
        The object to be checked.
    raise_err : bool, default False
        Whether to raise a ValueError if the object is not callable.

    Returns
    -------
    bool
        True if the object is callable, False otherwise.
    """
    if callable(obj):
        return True

    if raise_err:
        raise ValueError("The provided object must be callable.")

    return False

def is_lambda_func(obj: Any, raise_err_on_true: bool = False, raise_err_on_false: bool = False) -> bool:
    """
    Check whether a given object is a lambda function.

    Parameters
    ----------
    obj : Any
        The object to be checked.
    raise_err_on_true : bool, default False
        Whether to raise a ValueError if the object is a lambda function.
    raise_err_on_false : bool, default False
        Whether to raise a ValueError if the object is not a lambda function.

    Returns
    -------
    bool
        Whether the object is a lambda function.
    """
    if is_callable(obj, raise_err=raise_err_on_false):
        if obj.__name__ == "<lambda>":
            if raise_err_on_true:
                raise ValueError("The provided object cannot be a lambda function.")
            return True

        if raise_err_on_false:
            raise ValueError("The provided object must be a lambda function.")

    return False

def mutually_disjoint(*iterables: Iterable, raise_err: bool = False) -> bool:
    """
    Check whether a number of iterable values are mutually disjoint.

    Parameters
    ----------
    *iterables : iterable
        Variable number of iterable objects to check for mutual disjointness.
        Non-set iterables are converted to sets.
    raise_err : bool, default False
        Whether to raise an error if the iterable values are not mutually disjoint.

    Returns
    -------
    bool
        True if the iterable values are mutually disjoint, False otherwise.
    """
    # Convert non-set iterables to sets and use chain.from_iterable to combine them
    sets = [set(iterable) if not isinstance(iterable, set) else iterable for iterable in iterables]
    combined_set = set(chain.from_iterable(sets))
    
    # Check if the length of the combined set matches the sum of the individual set lengths
    if len(combined_set) == sum(len(s) for s in sets):
        return True

    if raise_err:
        raise ValueError("Values must be mutually disjoint.")

    return False

def exactly_one_specified(
    variables: Dict[str, Any],
    allow_none_specified: bool = False,
    raise_err: bool = False,
) -> bool:
    """
    Check that exactly one (or optionally no) variable(s) are being specified.

    Parameters
    ----------
    variables : dict
        A dictionary of variable names (keys) and values.
    allow_none_specified : bool, default False
        If True, allow all variables to be None without raising an error.
    raise_err : bool, default False
        If True, raise an error when the conditions are not met.

    Raises
    ------
    ValueError
        Number of specified variables is not 1 with allow_none_specified being False.
        Number of specified variables is not 0/1 with allow_none_specified being True.

    Returns
    -------
    bool
        True if the conditions are met, False otherwise.
    """
    specified_vars = [var for var, val in variables.items() if val is not None]
    
    if len(specified_vars) == 1:
        return True

    if len(specified_vars) == 0 and allow_none_specified:
        return True

    if raise_err:
        out_of_msg = f" out of: {', '.join(variables.keys())}. "
        if len(specified_vars) > 0:
            error_msg = f"Specified variables {', '.join(specified_vars)}" + out_of_msg
        else:
            error_msg = f"Did not specify any variables" + out_of_msg

        if allow_none_specified:
            error_msg = error_msg + "One or none of these variables should be specified."
        else:
            error_msg = error_msg + "Only one of these variables should be specified."

        raise ValueError(error_msg)

    return False
