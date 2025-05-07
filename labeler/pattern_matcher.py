from typing import List, Optional
import re
from dataclasses import dataclass, field

from tqdm import tqdm
tqdm.pandas()

from itertools import product

import numpy as np
import pandas as pd

from string_utils import border_alphanumeric, replace, unused_chars

class PatternMatcher:
    """
    Class to perform unified pattern matching while maintaining positional information.

    By replacing patterns with pattern IDs inplace, positional information is
    maintained and the same text cannot be matched twice.

    Each pattern is replaced with its string ID, where choice of replacement strings
    ensure that subsequent replacements do not disturb future replacements.

    Attributes
    ----------
    texts: pandas.Series
        A string series of the texts in which to identify patterns.
    patterns: pandas.Series
        A string series of the patterns to be matched, as sorted by reverse length to
        match on longer patterns before shorter ones.
    unusued_chars: list of str
        The 11 unused printable characters used in the `digit_map` and `delimiter`.
    digit_map: dict
        A map from digits 0-9 to various unused printable characters.
    delimiter: str
        An unused printable character used to delimit the the patterns.
    ids: pandas.Series
        A string series of pattern IDs, where the IDs are comprised of unused
        characters and each ID ends with the delimiter.
    pattern_to_id: dict
        Map from a pattern to its ID. IDs end with `delimiter`.
    id_to_pattern: dict
        Map from a pattern ID to its pattern. IDs do not include `delimiter`.
    """
    def __init__(self, texts: pd.Series, patterns: pd.Series):
        if not patterns.is_unique:
            vcs = patterns.value_counts()
            raise ValueError(
                f"Patterns must be unique:\n{str(vcs.index[vcs > 1])}"
            )

        self.texts = texts

        # Sort by reverse length to match on longer patterns before shorter ones
        self.patterns = patterns.reindex(
            patterns.str.len().sort_values(ascending=False).index
        )

        self.unused_chars = self._get_unused_chars()
        self.digit_map = dict(zip(range(10), self.unused_chars[:10]))
        self.delimiter = self.unused_chars[-1]

        self.ids = pd.Series(range(len(self.patterns))).astype('str')
        for num, replacement in self.digit_map.items():
            self.ids = self.ids.str.replace(
                str(num),
                replacement,
                regex=False,
            )

        self.pattern_to_id = dict(zip(self.patterns, self.ids + self.delimiter))
        self.id_to_pattern = dict(zip(self.ids, self.patterns))

    def _get_unused_chars(self):
        """
        Obtain printable characters which do not appear in the texts/patterns.
    
        Used to create the `digit_map` and `delimiter`.

        Returns
        -------
        list of str
            11 printable characters which do not appear in the texts/patterns.
        """
        # Get the set of characters used in the texts/patterns
        text_chars = set().union(
            *self.texts.apply(lambda string: set(string))
        ).union(
            *self.patterns.apply(lambda string: set(string))
        )

        # Get any printable characters not used in the texts
        chars = unused_chars(text_chars)

        if len(chars) < 11:
            raise ValueError(
                "texts cannot contain all printable characters: Must be 11 unused."
            )

        return list(chars)[:11]

    def __call__(
        self,
        batches: Optional[pd.DataFrame] = None,
        border_alnum: bool = True,
        progress: bool = True,
    ):
        """
        Support optional replacement batching, which is useful for when certain
        patterns should take priority over others, where patterns in earlier batches
        are prioritized over later batches.

        Parameters
        ----------
        batches: pandas.DataFrame
            Specify batches DataFrame to perform replacements in separate batches.
            Has an integer "batch" column and a string "pattern" column which
            contains all of the patterns.
        border_alnum: bool, default True
            Create alphanumeric borders.
        progress: bool, default True
            Show progress bar.
        
        Returns
        -------
        tuple of pandas.DataFrame
            Two DataFrames: One for text results/information and one for patterns.
            ----------------------
            Text DataFrame columns
            ----------------------
            text: The original text.
            replaced: Contains pattern IDs and unmatched text.
            unmatched: Contains only the unmatched text.
            stripped: The unmatched text stripped of whitespace and non-alphanumeric
            characters. Useful for determining how well the text was matched.
            pattern_id:  A series of ordered lists of pattern IDs.
            -------------------------
            Pattern DataFrame columns
            -------------------------
            pattern_id: Pattern IDs.
            pattern: Patterns.
            idx: Index to track the order of pattern appearence within each text.
        """
        if batches is None:
            # If no batches provided, create a stand-in batch for all patterns
            batches = self.patterns.copy().rename("pattern", inplace=True).to_frame()
            batches["batch"] = 0
        else:
            patterns_equal = batches["pattern"].sort_values().equals(
                batches["pattern"].sort_values()
            )
            if not patterns_equal:
                raise ValueError("Batch patterns are different from the patterns.")

        regex = border_alnum
        if border_alnum:
            pat = border_alphanumeric(self.patterns)
        else:
            pat = self.patterns

        replace_data = pd.DataFrame({
            "org": self.patterns, # An original pattern used to map to respective ID
            "pat": pat, # A possibly transformed regex pattern
        })

        text_results = pd.DataFrame({
            "text": self.texts.copy(),
            "replaced": self.texts.copy(),
        })

        for batch in sorted(batches["batch"].unique()):
            replace_subset = replace_data[
                replace_data["org"].isin(batches["pattern"][batches["batch"] == batch])
            ]

            iterator = replace_subset.iterrows()
            if progress:
                iterator = tqdm(iterator, total=len(replace_subset))

            for _, row in iterator:
                text_results["replaced"] = text_results["replaced"].str.replace(
                    row["pat"],
                    self.pattern_to_id[row["org"]],
                    regex=regex,
                )

        unused_chars_str = "".join(map(re.escape, self.unused_chars))

        text_results["unmatched"] = text_results["replaced"].str.replace(
            f"[{unused_chars_str}]",
            "",
            regex=True,
        )
        text_results["stripped"] = text_results["unmatched"].replace(
            "[^a-zA-Z0-9 ]",
            "",
            regex=True,
        ).str.strip()

        text_results["pattern_id"] = text_results["replaced"].str.replace(
            f"[^{unused_chars_str}]",
            "",
            regex=True,
        )

        # Split pattern ID text into separate patterns, but first strip the delimiter
        # since the last pattern leaves an unnecessary hanging delimiter
        pattern_ids = text_results["pattern_id"].str.strip(self.delimiter).str.split(
            self.delimiter,
        )

        # Explode the patterns to avoid dealing with slow lists
        pattern_ids.rename("pattern_id", inplace=True)
        pattern_results = pattern_ids.explode().dropna().to_frame()

        # Drop empty string entries introduced by samples with no patterns
        pattern_results = pattern_results[pattern_results["pattern_id"] != ""]

        pattern_results["idx"] = pattern_results.groupby(pattern_results.index).cumcount()
        pattern_results["pattern"] = pattern_results["pattern_id"].map(self.id_to_pattern)

        return text_results, pattern_results

@dataclass
class PatternReplacerResults:
    replaced: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "The texts with patterns replaced."},
    )
    text_results: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Pattern matching text-based results."},
    )
    pattern_results: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Pattern matching pattern-based results."},
    )
    id_repl: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Contains pattern, pattern ID, and replacement info."},
    )

class PatternReplacer(PatternMatcher):
    def __init__(self, texts: pd.Series, pat_repl: pd.DataFrame):
        super().__init__(texts, pat_repl["pat"])

        self.pat_repl = pat_repl

    def __call__(
        self,
        **matcher_kwargs,
    ):
        text_results, pattern_results = super().__call__(**matcher_kwargs)

        # Turn pattern ID to pattern mapping into a dictionary
        id_repl = pd.DataFrame(self.id_to_pattern.items(), columns=["id", "pat"])

        # Add in the pattern replacements via a merge
        id_repl = id_repl.merge(self.pat_repl, on="pat")

        # Add in the delimiter character to make sure it is removed
        id_repl["id"] = id_repl["id"] + self.delimiter

        # Replace pattern IDs with the desired replacements
        replaced, _ = replace(
            text_results["replaced"].copy(),
            id_repl.drop("pat", axis=1).rename({"id": "pat"}, axis=1),
            border_alnum=False, # No need to border pattern IDs
        )

        return PatternReplacerResults(
            replaced=replaced,
            text_results=text_results,
            pattern_results=pattern_results,
            id_repl=id_repl,
        )

@dataclass
class ConjunctResult:
    template: str
    group_nums: List[int]
    product: List[str]
    pattern: str
    product_mapped: List[str]
    pattern_mapped: str
    pattern_replaced: str

class ConjunctPatterns:
    def __init__(
        self,
        groups: List[List[str]],
        templates: List[str],
        ignore_dups: bool = False,
    ):
        """
        Identify and encode patterns.

        Create a number of patterns and corresponding replacement patterns.

        Standardize multiple ways of saying the same thing.

        For example,
        Template: "four [0] and [2] years ago: "[2] years"
        Original: "four score and sixty years ago"
        Replaced: "sixty years"

        Parameters
        ----------

        ignore_dups: bool, default False
            Ignore duplicate patterns, provided they have the same replacement string.
        """
        # Convert group lists to dictionaries mapping each value to itself
        for i, group in enumerate(groups):
            if not isinstance(group, dict):
                groups[i] = dict(zip(group, group))

        self.groups = groups
        
        if not isinstance(templates, dict):
            self.templates = dict(zip(templates, [None]*len(templates)))
        else:
            self.templates = templates

        self.group_products = {}
        self.results = None
        self.ignore_dups = ignore_dups

    def _extract_nums(self, string: str) -> np.ndarray:
        """
        Extract group numbers, e.g., `[0, 1, 0]` from: "A [0] is a [1] to [0]."
        """
        return np.array(re.findall(r"\[(\d+)\]", string)).astype(int)

    def _group_nums(self, string: str) -> np.ndarray:
        """
        Get group numbers from the patterns.
        """
        group_nums = self._extract_nums(string)

        if len(group_nums) == 0:
            return group_nums

        if group_nums.max() >= len(self.groups):
            raise ValueError(
                f"String '{string}' contains invalid group number {group_nums.max()}."
            )

        return group_nums

    def _repl_group_nums(self, string: str, group_nums: np.ndarray):
        """
        Get numbers from the replacements.
        """
        repl_group_nums = self._extract_nums(string)

        if (repl_group_nums > len(group_nums) - 1).any():
            raise ValueError(
                f"String '{string}' contains invalid replacement group number > "
                f"{len(group_nums) - 1}."
            )

        return repl_group_nums

    def __call__(
        self,
    ):
        results = []
        for tem, tem_repl in self.templates.items():
            group_nums = self._group_nums(tem)
            key = '-'.join(group_nums.astype(str))

            if key not in self.group_products:
                self.group_products[key] = list(
                    product(*[self.groups[i] for i in group_nums])
                )

            for prod in self.group_products[key]:
                prod = list(prod)
                mapped_prod = [self.groups[group_nums[i]][p] for i, p in enumerate(prod)]
                pat = tem
                mapped_pat = tem
                for i, num in enumerate(group_nums):
                    pat = pat.replace(f"[{num}]", prod[i], 1)
                    mapped_pat = mapped_pat.replace(f"[{num}]", mapped_prod[i], 1)

                # Prepare replaced pattern
                if tem_repl is None:
                    # If no template replacement, keep the existing mapped pattern
                    pat_repl = mapped_pat
                else:
                    # Otherwise, create the replaced pattern
                    repl_group_nums = self._repl_group_nums(tem_repl, group_nums)

                    # Simply use the template replacement if it is a constant string
                    pat_repl = tem_repl
                    if len(repl_group_nums) != 0:
                        for i, num in enumerate(repl_group_nums):
                            pat_repl = pat_repl.replace(f"[{num}]", mapped_prod[num])

                results.append(ConjunctResult(
                    template = tem,
                    group_nums = group_nums,
                    product = prod,
                    pattern = pat,
                    product_mapped = mapped_prod,
                    pattern_mapped = mapped_pat,
                    pattern_replaced = pat_repl,
                ))

        self.results = results

        return results

    def replace(self, texts: pd.Series, **matcher_kwargs):
        if self.results is None:
            raise ValueError("Must first call before replacing.")

        # Perform pattern replacements
        pat_repl = pd.DataFrame({
            "pat": [result.pattern for result in self.results],
            "repl": [result.pattern_replaced for result in self.results],
        })

        if self.ignore_dups:
            # Drop duplicates having the same pattern and replacement pattern; Can't
            # only drop pattern duplicates - how do we know which replacement to use?
            pat_repl = pat_repl.drop_duplicates()

        self.patrepl = PatternReplacer(texts, pat_repl)

        return self.patrepl(**matcher_kwargs)
