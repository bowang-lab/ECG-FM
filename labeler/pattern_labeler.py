"""
TODO

Check the new templates for non-existent entities and descriptors
- Make this its own function

Add static in-class standardizing functionalities (to template, entity, etc.)
"""
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import os
import re
from dataclasses import dataclass, field
import inspect
from functools import wraps
from copy import deepcopy
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

from common import flatten_2d_list, to_list, to_list_optional
from pattern_matcher import PatternMatcher

class DescriptorTem:
    descriptors = [] # Unused, besides improve type invariance amongst EntityTem/Entity

    def __init__(
        self,
        name,
        category = None,
        attribute_to: Optional[Union[str, List[str]]] = None,
        tags: Optional[Union[str, List[str]]] = None,
        create_entities: Optional[Union[str, List[str]]] = None,
        patterns: Optional[Union[str, List[str]]] = None,
    ):
        """
        attribute_to: list of str, optional
            A list of entities to which this descriptor is allowed to attribute. If
            None, it can attribute to any entity.
        """
        self.name = name
        self.category = category
        self.attribute_to = to_list_optional(attribute_to)
        self.tags = to_list_optional(tags, none_to_empty=True)
        self.create_entities = to_list_optional(create_entities, none_to_empty=True)
        self.patterns = to_list_optional(patterns, none_to_empty=True)

    def __repr__(self):
        return self.name

    def to_instance(self):
        return Descriptor(self)

    def strings_to_templates(self, entity_template_map, descriptor_template_map):
        if self.attribute_to is not None:
            self.attribute_to = [entity_template_map[entity] for entity in self.attribute_to]

        for ind, entity in enumerate(self.create_entities):
            if isinstance(entity, str):
                self.create_entities[ind] = entity_template_map[entity]

    def split_tem_to_tem(self):
        pass

class SplitDescriptorTem:
    def __init__(
        self,
        name: str,
        split: List[DescriptorTem],
        patterns=None,
    ):
        """
        A descriptor which can be represented as a combination of other descriptors.
        E.g., "Anterolateral" can be represented as "Anterior" and "Lateral".
        """
        self.name = name
        self.split = split
        self.patterns = to_list_optional(patterns, none_to_empty=True)

    def __repr__(self):
        return self.name

    def strings_to_templates(self, entity_template_map, descriptor_template_map):
        for ind, descriptor in enumerate(self.split):
            if isinstance(descriptor, str):
                self.split[ind] = descriptor_template_map[descriptor]

    def split_tem_to_tem(self):
        pass

def split_descriptors(descriptors):
    # Split into descriptors
    for descriptor in descriptors:
        if isinstance(descriptor, SplitDescriptorTem):
            descriptors.extend(descriptor.split)

    # Remove split descriptors
    return [desc for desc in descriptors if not isinstance(desc, SplitDescriptorTem)]

class Descriptor:
    def __init__(self, tem: Union[DescriptorTem, str], descriptors = None):
        """
        descriptors : 
            E.g., for 'probably recent block', self.descriptors=['Probably']
        """
        self.tem = tem
        self.descriptors = to_list_optional(
            descriptors,
            none_to_empty=True,
        )

    def __repr__(self):
        if isinstance(self.tem, str):
            return self.tem

        return self.tem.name

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    def add_descriptors(self, descriptors):
        for descriptor in to_list(descriptors):
            self.add_descriptor(descriptor)

    def to_instance(self):
        return deepcopy(self)

    def split_tem_to_tem(self):
        self.descriptors = split_descriptors(self.descriptors)

DESCRIPTOR_TYPES = Union[DescriptorTem, SplitDescriptorTem]
DESCRIPTOR_LIST_TYPE = Union[DESCRIPTOR_TYPES, List[DESCRIPTOR_TYPES]]

class DescriptorPattern:
    """
    parse_before_entities : bool, default False
        E.g., "in a pattern of bigeminy" must be parsed before a "bigeminy" EntityPattern.
    """
    def __init__(
        self,
        pattern: str,
        descriptors: DESCRIPTOR_TYPES,
        before_after: Optional[str] = None,
        parse_before_entities: bool = False,
    ):
        self.pattern = pattern
        self.descriptors = to_list(descriptors)

        if before_after is None:
            self.before_after = "b"
        else:
            assert before_after in ["a", "b", "ab"]
            self.before_after = before_after

        self.parse_before_entities = parse_before_entities

    def __repr__(self):
        return self.pattern

    def strings_to_templates(self, descriptor_template_map):
        for ind, descriptor in enumerate(self.descriptors):
            if isinstance(descriptor, str):
                self.descriptors[ind] = descriptor_template_map[descriptor]

    def split_tem_to_tem(self):
        self.descriptors = split_descriptors(self.descriptors)

class EntityTem:
    descriptors = [] # Unused, besides improve type invariance amongst EntityTem/Entity

    def __init__(
        self,
        name: str,
        sup: Optional[Union[str, List[str]]] = None,
        aka: Optional[Union[str, List[str]]] = None,
        require_descriptors: bool = False,
        tags: Optional[Union[str, List[str]]] = None,
        #patterns: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        """
        require_descriptors : bool, default False
            Whether an entity requires a descriptor to be considered a label, where
            some entities only really only relevant when a descriptor is present.
        """
        self.name = name
        self.sup = to_list_optional(sup, none_to_empty=True)
        self.aka = to_list_optional(aka, none_to_empty=True)
        self.require_descriptors = require_descriptors
        self.tags = to_list_optional(tags, none_to_empty=True)
        #self.patterns = to_list_optional(patterns, none_to_empty=True)

        for key, value in kwargs.items():
            setattr(self, key, to_list_optional(value, none_to_empty=True))

    def __repr__(self):
        return self.name

    def to_instance(self):
        return Entity(self)

    def strings_to_templates(self, entity_template_map, descriptor_template_map):
        self.sup = [entity_template_map[entity] for entity in self.sup]

    def split_tem_to_tem(self):
        pass

class Entity:
    def __init__(self, tem: Union[EntityTem, str], descriptors = None):
        self.tem = tem
        self.descriptors = to_list_optional(
            descriptors,
            none_to_empty=True,
        )

    def __repr__(self):
        if isinstance(self.tem, str):
            return self.tem

        return self.tem.name

    def add_descriptor(self, descriptor, skip_if_exists: bool = False):
        if not skip_if_exists:
            self.descriptors.append(descriptor)
            return

        # Skip adding this descriptor if it already exists
        if any(descriptor.name == desc.name for desc in self.descriptors):
            return

        self.descriptors.append(descriptor)

    def add_descriptors(self, descriptors, skip_if_exists: bool = False):
        for descriptor in to_list(descriptors):
            self.add_descriptor(descriptor, skip_if_exists=skip_if_exists)

    def to_instance(self):
        return deepcopy(self)

    def split_tem_to_tem(self):
        self.descriptors = split_descriptors(self.descriptors)

ENTITIES_TYPES = Entity
ENTITY_LIST_TYPE = Union[ENTITIES_TYPES, List[ENTITIES_TYPES]]

class EntityPattern:
    """
    entities
        Entities matched by the pattern. If None, then the pattern doesn't match to
        any entities and is essentially ignored.
    """
    def __init__(
        self,
        pattern: str,
        entities: Optional[ENTITY_LIST_TYPE],
    ):
        self.pattern = pattern
        self.entities = to_list_optional(entities, none_to_empty=True)

    def __repr__(self):
        return self.pattern

    def add_descriptor(self, descriptor: Descriptor):
        for entity in self.entities:
            entity.add_descriptor(descriptor)

    def add_descriptors(self, descriptors: List[Descriptor]):
        for descriptor in to_list(descriptors):
            self.add_descriptor(descriptor)

    def to_instance(self):
        self.entities = [entity.to_instance() for entity in self.entities]
        return self

    def strings_to_templates(self, entity_template_map):
        for ind, entity in enumerate(self.entities):
            if isinstance(entity, str):
                self.entities[ind] = entity_template_map[entity]

    def split_tem_to_tem(self):
        for entity in self.entities:
            if isinstance(entity, Entity):
                entity.split_tem_to_tem()

class Connective:
    def __init__(
        self,
        pattern: str,
        descriptors: Optional[Tuple[
            Optional[DESCRIPTOR_LIST_TYPE],
            Optional[DESCRIPTOR_LIST_TYPE],
        ]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.pattern = pattern
        if descriptors is None:
            self.descriptors = [[], []]
        else:
            self.descriptors = [
                to_list_optional(descriptors[0], none_to_empty=True),
                to_list_optional(descriptors[1], none_to_empty=True),
            ]
        self.tags = to_list_optional(tags)

    def __repr__(self):
        return self.pattern

    def strings_to_templates(self, descriptor_template_map):
        for ind, descriptor in enumerate(self.descriptors[0]):
            if isinstance(descriptor, str):
                self.descriptors[0][ind] = descriptor_template_map[descriptor]

        for ind, descriptor in enumerate(self.descriptors[1]):
            if isinstance(descriptor, str):
                self.descriptors[1][ind] = descriptor_template_map[descriptor]

    def split_tem_to_tem(self):
        self.descriptors = split_descriptors(self.descriptors)

class CompoundTem:
    """
    If the `including` entity has descriptors, then those must been included,
    but don't have to be the only descriptors.
    """
    def __init__(
        self,
        entity, #: Union[str, Entity],
        including,  #: List[Union[[str, Entity]]],
        excluding=None, #: Optional[List[Union[[str, Entity]]]] = None,
    ):
        self.entity = entity
        self.including = to_list_optional(including, none_to_empty=True)
        self.excluding = to_list_optional(excluding, none_to_empty=True)

    def __repr__(self):
        string = str(self.entity) + \
            f" (+ {', '.join([str(entity) for entity in self.including])})"
        if len(self.excluding) > 0:
            string += f" (- {', '.join([str(entity) for entity in self.excluding])})"

        return string

    def strings_to_instances(self, entity_template_map, descriptor_template_map):
        self.entity = entity_template_map[self.entity].to_instance() if \
            isinstance(self.entity, str) else self.entity
        self.including = [
            entity_template_map[entity].to_instance() if isinstance(entity, str) \
                else entity for entity in self.including
        ]
        self.excluding = [
            entity_template_map[entity].to_instance() if isinstance(entity, str) \
                else entity for entity in self.excluding
        ]

class AttachedDescriptorTem:
    def __init__(
        self,
        resulting_entity: str,
        entity: str,
        descriptor: str,
    ):
        self.resulting_entity = resulting_entity
        self.entity = entity
        self.descriptor = descriptor

    def __repr__(self):
        return f"{self.entity} - {self.descriptor} ({self.resulting_entity})"

class TravelingDescriptorsTem:
    def __init__(self, entity: str, descriptors: Union[str, List[str]]):
        self.entity = entity
        self.descriptors = to_list(descriptors)

    def __repr__(self):
        return f"Descriptors {self.descriptors} travel to {self.entity}"

@dataclass
class PatternLabelerResults:
    texts: Optional[pd.Series] = field(
        default=None,
        metadata={"help": "Original texts."},
    )
    text_results: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Pattern matching text-based results."},
    )
    pattern_results: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Pattern matching pattern-based results."},
    )
    entities_text: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Entities extracted from the text (no manipulation)."},
    )
    descriptors_text: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Descriptors extracted from the text (no manipulation)."},
    )
    entities: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Entities."},
    )
    descriptors: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Descriptors."},
    )
    labels: Optional[pd.DataFrame] = field(
        default=None,
        metadata={
            "help": "Labels. May include duplicates labels with differing probabilities."
        },
    )
    labels_flat: Optional[pd.DataFrame] = field(
        default=None,
        metadata={"help": "Final labels."},
    ) 

def labels_to_array(
    labels: pd.DataFrame,
    num_samples: int,
    label_order: Optional[List[str]] = None,
):
    # Create ordered label definition DataFrame to track label information
    label_def = pd.Series(labels["name"].unique())

    if label_order is None:
        # Order alphabetically
        label_def.sort_values(inplace=True)
    else:
        # Use a provided order
        assert set(label_def) == set(label_order)
        label_def = pd.Series(label_order)

    label_def = label_def.rename("name").to_frame()
    label_def.reset_index(drop=True, inplace=True)

    # Add label ID into labels
    labels_merged = labels.merge(
        label_def.reset_index().rename({"index": "id"}, axis=1)[["name", "id"]],
        on="name",
        how="left",
    )

    # Restore index after merge
    labels_merged.index = labels.index
    labels = labels_merged
    labels.index.name = "idx"

    # Convert from labels DataFrame into soft labels array
    index = np.arange(num_samples)
    y_soft = []
    for label_id in sorted(labels["id"].unique()):
        partition = labels[
            labels["id"] == label_id
        ]["prob"]
        y_soft.append(partition.reindex(index, fill_value=0).values)

    y_soft = np.stack(y_soft, axis=1)

    # Compute hard labels from soft labels
    y = (y_soft > 0).astype('float32')

    # Compute more label information
    label_def["count"] = y.sum(axis=0).astype(int)
    label_def["pos_percent"] = label_def["count"]/len(y)

    # pos weight = num negative / num positive = (1 - pos percent) / pos percent
    label_def["pos_weight"] = (1 - label_def["pos_percent"]) / label_def["pos_percent"]

    return labels, label_def, y_soft, y

def create_ancestor_matrix(graph: nx.DiGraph, nodes = None) -> (np.ndarray, dict):
    """
    Creates a 2D boolean matrix indicating if each node in a DAG is an ancestor of another node and
    a dictionary mapping node names to their corresponding indices in the matrix.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed acyclic graph (DAG).

    Returns
    -------
    np.ndarray
        A 2D boolean matrix where matrix[i][j] is True if node i is an ancestor of node j.
    dict
        A dictionary mapping node names to their corresponding indices in the matrix.

    Raises
    ------
    ValueError
        If the input graph is not a DAG.
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The input graph must be a DAG.")

    if nodes is None:
        nodes = list(graph.nodes)

    node_index = {node: idx for idx, node in enumerate(nodes)}
    ancestor_matrix = np.zeros((len(nodes), len(nodes)), dtype=bool)

    for node in nodes:
        descendants = nx.descendants(graph, node)
        for descendant in descendants:
            ancestor_matrix[node_index[node], node_index[descendant]] = True

    return ancestor_matrix, node_index

def pickle_on_error_with_provided_path(
    attribute: str,
    restore_path_arg: str,
    pickle_on_return: bool = False,
):
    """
    A decorator to pickle an object's attribute to a file dynamically specified
    in the method arguments if an exception occurs or upon successful return.

    Parameters
    ----------
    attribute : str
        Name of the attribute to pickle.
    restore_path_arg : str
        Name of the method argument specifying the file path.
    pickle_on_return : bool, default False
        Whether to pickle the attribute when the function completes successfully.

    Returns
    -------
    function
        Wrapped function with error handling and optional pickling on return.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                if pickle_on_return:
                    # Extract the file path from arguments
                    sig = inspect.signature(func)
                    bound_args = sig.bind(self, *args, **kwargs)
                    bound_args.apply_defaults()
                    save_path = bound_args.arguments.get(restore_path_arg, None)

                    if save_path:
                        attr_value = getattr(self, attribute, None)
                        if attr_value is not None:
                            with open(save_path, "wb") as f:
                                pickle.dump(attr_value, f)
                return result
            except Exception as err:
                # Extract the file path from arguments
                sig = inspect.signature(func)
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                save_path = bound_args.arguments.get(restore_path_arg, None)

                if save_path:
                    attr_value = getattr(self, attribute, None)
                    if attr_value is not None:
                        with open(save_path, "wb") as f:
                            pickle.dump(attr_value, f)
                raise err  # Re-raise the exception
        return wrapper
    return decorator

@dataclass
class PatternLabelerConfig:
    ARGUMENTS_REQUIRED = OrderedDict([
        ('entity_tem', True),
        ('entity_pat', True),
        ('desc_tem', False),
        ('desc_pat', False),
        ('split_desc_tem', False),
        ('connective', False),
        ('compound_tem', False),
        ('attached_desc_tem', False),
        ('traveling_desc_tem', False),
        ('uncertainty_map', False),
    ])
    entity_templates: list
    entity_patterns: list
    descriptor_templates: Optional[List[str]] = field(default_factory=list)
    descriptor_patterns: Optional[List[str]] = field(default_factory=list)
    split_descriptor_templates: Optional[List[str]] = field(default_factory=list)
    connectives: Optional[List[str]] = field(default_factory=list)
    compound_templates: Optional[List[str]] = field(default_factory=list)
    attached_descriptor_templates: Optional[List[str]] = field(default_factory=list)
    traveling_descriptor_templates: Optional[List[str]] = field(default_factory=list)
    uncertainty_map: Optional[Dict[str, float]] = None
    progress: bool = True

    def to_json(self, save_directory: str):
        from to_json import TO_JSON_FUNC
        os.makedirs(save_directory, exist_ok=True)

        save_objs = {
            'entity_tem': self.entity_templates,
            'entity_pat': self.entity_patterns,
            'desc_tem': self.descriptor_templates,
            'desc_pat': self.descriptor_patterns,
            'split_desc_tem': self.split_descriptor_templates,
            'connective': self.connectives,
            'compound_tem': self.compound_templates,
            'attached_desc_tem': self.attached_descriptor_templates,
            'traveling_desc_tem': self.traveling_descriptor_templates,
        }
        if self.uncertainty_map is not None:
            save_objs['uncertainty_map'] = self.uncertainty_map

        for name, obj in save_objs.items():
            if obj is None:
                continue

            if isinstance(obj, list) and len(obj) == 0:
                continue

            TO_JSON_FUNC[name](save_directory, deepcopy(obj))

    @classmethod
    def from_json(
        cls,
        directory: str,
        progress: bool = True,
    ):
        from from_json import FROM_JSON_FUNC

        args = OrderedDict()
        for arg, required in cls.ARGUMENTS_REQUIRED.items():
            json_path = os.path.join(directory, f'{arg}.json')

            if os.path.isfile(json_path):
                args[arg] = FROM_JSON_FUNC[arg](directory)
            else:
                if required:
                    raise ValueError(f'{arg}.json is required.')

                if arg == 'uncertainty_map':
                    args[arg] = None
                else:
                    args[arg] = []

        return cls(*args.values(), progress=progress)

class PatternLabeler:
    def __init__(self, config: PatternLabelerConfig):
        self.entity_templates = config.entity_templates
        self.entity_patterns = config.entity_patterns
        self.descriptor_templates = \
            config.descriptor_templates + config.split_descriptor_templates
        self.descriptor_patterns = config.descriptor_patterns
        self.connectives = config.connectives
        self.compound_templates = config.compound_templates
        self.attached_descriptor_templates = config.attached_descriptor_templates
        self.traveling_descriptor_templates = config.traveling_descriptor_templates
        self.uncertainty_map = config.uncertainty_map
        self.progress = config.progress

        self.entity_template_map = {tem.name: tem for tem in self.entity_templates}
        self.descriptor_template_map = {tem.name: tem for tem in self.descriptor_templates}

        # Preprocess the data
        self._strings_to_templates()
        self._extract_patterns()
        self._split_tem_to_tem()

        # Ensure entity names are unique
        self.entity_names = [entity.name for entity in self.entity_templates]
        if len(set(self.entity_names)) != len(self.entity_names):
            print(pd.Series(self.entity_names).value_counts())
            raise ValueError("Entity names must be unique.")

        # Ensure descriptor names are unique
        self.descriptor_names = [descriptor.name for descriptor in self.descriptor_templates]
        if len(set(self.descriptor_names)) != len(self.descriptor_names):
            print(pd.Series(self.descriptor_names).value_counts())
            raise ValueError("Descriptor names must be unique.")

        # Compile and check all patterns
        self.patterns = \
            self.entity_patterns + \
            self.descriptor_patterns + \
            self.connectives
        self.pattern_names = [pattern.pattern for pattern in self.patterns]
        self.name_to_pattern = dict(zip(self.pattern_names, self.patterns))

        # Ensure all pattern names are unique
        if not len(set(self.pattern_names)) == len(self.patterns):
            print(pd.Series(
                [pattern.pattern for pattern in self.patterns]
            ).value_counts().iloc[:60])
            raise ValueError("Patterns must be unique.")

        # Create a graph which is useful for perform several parsing checks/operations
        self.sup_graph = nx.DiGraph()
        for entity in self.entity_templates:
            self.sup_graph.add_node(str(entity.name))
            for sup in entity.sup:
                self.sup_graph.add_edge(str(entity.name), str(sup))

        # Assert that we have a DAG - acyclic graph - otherwise the parsing falls apart
        cycles = list(nx.simple_cycles(self.sup_graph))
        if len(cycles) != 0:
            raise ValueError(f"Cycles found: {cycles}")

        # Create a 2D lookup table to check whether an entity is an ancestor of another
        self.ancestor_matrix, self.entity_idx_map = create_ancestor_matrix(
            self.sup_graph,
            nodes=pd.Series(self.entity_templates).apply(str).values
        )
        
        self.sup_map = self._create_sup_map()

        self.res = None

    def _create_sup_map(self):
        sup_map = pd.Series(
            self.entity_templates,
            index=self.entity_templates,
        ).apply(lambda tem: tem.sup).explode().dropna()
        sup_map = sup_map.rename("sup")
        sup_map.index.name = "name"
        sup_map = sup_map.reset_index()
        sup_map["name"] = sup_map["name"].astype(str)
        sup_map["sup"] = sup_map["sup"].astype(str)
        sup_map.index = sup_map["name"].map(self.entity_idx_map)
        sup_map.index.name = None

        return sup_map

    def _split_tem_to_tem(self):
        # Split SplitDescriptorTem into multiple DescriptorTem and remove them all
        for template in self.descriptor_templates:
            template.split_tem_to_tem()

        self.descriptor_templates = [tem for tem in self.descriptor_templates if not isinstance(tem, SplitDescriptorTem)]

        for template in self.entity_templates:
            template.split_tem_to_tem()

        for pattern in self.descriptor_patterns:
            pattern.split_tem_to_tem()

        for pattern in self.connectives:
            pattern.split_tem_to_tem()

        for pattern in self.entity_patterns:
            pattern.split_tem_to_tem()

    def _strings_to_templates(self):
        # Convert string instances to templates
        for template in self.descriptor_templates:
            assert isinstance(template, DescriptorTem) or isinstance(template, SplitDescriptorTem)
            template.strings_to_templates(self.entity_template_map, self.descriptor_template_map)

        for template in self.entity_templates:
            assert isinstance(template, EntityTem)
            template.strings_to_templates(self.entity_template_map, self.descriptor_template_map)

        for pattern in self.descriptor_patterns:
            assert isinstance(pattern, DescriptorPattern)
            pattern.strings_to_templates(self.descriptor_template_map)

        for pattern in self.connectives:
            assert isinstance(pattern, Connective)
            pattern.strings_to_templates(self.descriptor_template_map)

        for pattern in self.entity_patterns:
            assert isinstance(pattern, EntityPattern)
            pattern.strings_to_templates(self.entity_template_map)

    def _extract_patterns(self):
        # Extract patterns defined directly in the Entity/Descriptor objects
        # for tem in self.entity_templates:
        #     for pattern in tem.patterns:
        #         self.entity_patterns.append(EntityPattern(pattern, tem))

        for tem in self.descriptor_templates:
            for pattern in tem.patterns:
                self.descriptor_patterns.append(DescriptorPattern(pattern, tem))

    def _match(self, texts: pd.Series):
        """
        Perform pattern matching.

        Creates batches in order of pattern matching importance:
         - Descriptor patterns marked to be parsed before entities
         - Entities patterns
         - Remaining descriptor patterns
         - Connective patterns
        
        """
        print("Matching patterns...")

        # Create pattern-matching batches
        pattern_batches = [
            [
                pattern.pattern for pattern in self.descriptor_patterns \
                    if pattern.parse_before_entities
            ],
            [pattern.pattern for pattern in self.entity_patterns],
            [
                pattern.pattern for pattern in self.descriptor_patterns \
                    if not pattern.parse_before_entities
            ],
            [connective.pattern for connective in self.connectives],
        ]
        for i, batch in enumerate(pattern_batches):
            batch = pd.Series(batch, dtype=str).rename("pattern", inplace=True).to_frame()
            batch["batch"] = i
            pattern_batches[i] = batch

        pattern_batches = pd.concat(pattern_batches)

        # Perform pattern matching
        self.patmat = PatternMatcher(texts, pd.Series(self.pattern_names))

        return self.patmat(batches=pattern_batches, progress=self.progress)

    def _extract_entities(self, pattern_results: pd.DataFrame):
        """
        Create entities, attributing descriptors from descriptors and connectives.
        """
        # Attribute descriptors/connectives to patterns
        print("Creating entities and attributing descriptors...")

        def map_to_letter(pattern):
            if isinstance(pattern, EntityPattern) or isinstance(pattern, Entity):
                return "e"
            if isinstance(pattern, Connective):
                return "c"
            if isinstance(pattern, DescriptorPattern):
                if pattern.before_after == "b":
                    return "b"
                if pattern.before_after == "a":
                    return "a"

        def attributions(types: np.ndarray):
            attributions = []

            # Find the indices of "e" occurrences
            e_indices = [i for i, letter in enumerate(types) if letter == 'e']

            for i, letter in enumerate(types):
                if letter == 'c':
                    # Find the closest left and right "e" indices
                    left_e_index = max([e_index for e_index in e_indices if e_index < i] or [-1])
                    right_e_index = min([e_index for e_index in e_indices if e_index > i] or [len(types)])

                    # Attribute the "c" to the closest left and right "e"
                    if left_e_index == -1 and right_e_index == len(types):
                        attributions.append((np.nan, np.nan))
                    elif left_e_index == -1:
                        attributions.append((np.nan, right_e_index))
                    elif right_e_index == len(types):
                        attributions.append((left_e_index, np.nan))
                    else:
                        attributions.append((left_e_index, right_e_index))

                elif letter == 'b':
                    # Find the closest right "e" index
                    right_e_index = min([e_index for e_index in e_indices if e_index > i] or [len(types)])
                    if right_e_index == len(types):
                        attributions.append(np.nan)
                    else:
                        attributions.append(right_e_index)

                elif letter == 'a':
                    # Find the closest left "e" index
                    left_e_index = max([e_index for e_index in e_indices if e_index < i] or [-1])
                    if left_e_index == -1:
                        attributions.append(np.nan)
                    else:
                        attributions.append(left_e_index)

                else:
                    attributions.append(np.nan)

            return attributions

        pattern_results = pattern_results.rename({"pattern": "pattern_str"}, axis=1)
        pattern_results["pattern"] = pattern_results["pattern_str"].map(
            self.name_to_pattern
        )
        pattern_results["type"] = pattern_results["pattern"].apply(map_to_letter)

        def parse_patterns(group: pd.DataFrame):
            group = group.set_index("idx")

            # Get indices of the entities
            e_indices = group.index[group['type'] == 'e']

            if len(e_indices) == 0:
                return group

            # Determine how everything attributes to each entity
            group["attribute"] = attributions(group['type'].values)

            # Determine the attributions to each entity
            group["attribution"] = np.empty((len(group), 0)).tolist()
            for index, val in group["attribute"].items():
                if isinstance(val, int):
                    group["attribution"].loc[val].append(index)

                if isinstance(val, tuple):
                    if not pd.isna(val[0]):
                        group["attribution"].loc[val[0]].append((index, 0))

                    if not pd.isna(val[1]):
                        group["attribution"].loc[val[1]].append((index, 1))

            # Use entity patterns and attributions to create the entities
            group["entity"] = np.empty((len(group), 0)).tolist()
            for idx in e_indices:
                row = group.loc[idx]
                entity_pattern = row["pattern"]

                descriptors = []

                if row["attribution"] != np.nan:
                    for atr in row["attribution"]:
                        if isinstance(atr, int):
                            # Add descriptors from a descriptor pattern
                            descriptors.extend(group["pattern"].loc[atr].descriptors)
                        else:
                            # Add descriptors from a connective pattern
                            descriptors.extend(
                                group["pattern"].loc[atr[0]].descriptors[atr[1]]
                            )

                for entity in entity_pattern.entities:
                    entity_tem = entity if isinstance(entity, EntityTem) else entity.tem
                    row["entity"].append(
                        Entity(entity_tem, descriptors=entity.descriptors + descriptors)
                    )

            return group

        print("Parsing patterns into entities...")
        pattern_results = pattern_results.groupby(pattern_results.index).progress_apply(
            parse_patterns
        )

        pattern_results.reset_index(inplace=True)
        pattern_results.set_index("level_0", inplace=True)
        pattern_results.index.name = None
        pattern_results = pattern_results.map(
            lambda x: np.nan if isinstance(x, list) and not x else x,
        )

        # Explode and drop those NaN rows representing texts without entities
        entities = pattern_results["entity"].explode().dropna()

        # Standardize by converting descriptor templates into descriptors
        def descriptor_templates_to_descriptors(entity):
            for ind, desc in enumerate(entity.descriptors):
                if isinstance(desc, DescriptorTem):
                    entity.descriptors[ind] = Descriptor(desc)

        entities.apply(descriptor_templates_to_descriptors)

        # Add in sample-specific entity IDs into the index (in order of appearence)
        entities = entities.rename("entity").to_frame()
        entities = entities.set_index(entities.groupby(entities.index).cumcount(), append=True)

        # Make unique identifiers for the sample, pattern, and entity idx
        entities = entities.set_index(entities.groupby(entities.index).cumcount(), append=True)
        entities.index.names =["sample", "pattern", "entity"]

        # Extract descriptors data
        descriptors = entities["entity"].apply(
            lambda entity: entity.descriptors
        ).explode().dropna().astype(str).rename("name").to_frame()

        # Convert entities to strings (no other useful data)
        entities["name"] = entities["entity"].astype(str)
        del entities["entity"]

        return pattern_results, entities, descriptors

    def incorporate_new_entities(
        self,
        entities,
        entities_add,
        extract_super: bool = False,
        index_in_source: bool = True,
    ):
        if index_in_source:
            entities_add["source"] += " [" + entities_add.index.map(str) + "]"

        if entities_add.index.nlevels == 1:
            # If only "sample" is provided, then create an appropriate "pattern" level
            assert entities_add.index.names == ["sample"]

            entities_subset = entities[entities.index.get_level_values(0).isin(entities_add.index)]
            next_available_pattern = entities_subset.reset_index(level=[1, 2]).groupby(
                level=0,
            )['pattern'].max() + 1

            entities_add["pattern"] = entities_add.groupby(entities_add.index).cumcount()
            entities_add["pattern"] += next_available_pattern.loc[entities_add.index]
            entities_add.set_index("pattern", append=True, inplace=True)

            entities_add["entity"] = entities_add.groupby(entities_add.index).cumcount()
            entities_add.set_index("entity", append=True, inplace=True)

        else:
            if entities_add.index.nlevels == 2:
                assert entities_add.index.names == ["sample", "pattern"]

            elif entities_add.index.nlevels == 3:
                # Drop entity so that it can be arranged correctly
                assert entities_add.index.names == ["sample", "pattern", "entity"]
                entities_add.reset_index(level=2, drop=True, inplace=True)

            # Create an appropriate "entity" level
            next_available_entity = entities.reset_index(level=2).groupby(
                level=[0, 1],
            )['entity'].max() + 1

            entities_add["entity"] = entities_add.groupby(entities_add.index).cumcount()
            entities_add["entity"] += next_available_entity.loc[entities_add.index]
            entities_add.set_index("entity", append=True, inplace=True)

        if extract_super:
            entities_add = self._extract_super_entities(entities_add)

        return pd.concat([entities, entities_add]).sort_index()

    def _extract_desc_entities(self, entities, descriptors):
        """
        Create entities from a DescriptorTem's `create_entities` attribute.
        """
        desc_entities = descriptors["name"].map(self.descriptor_template_map).apply(
            lambda tem: tem.create_entities
        ).explode().dropna().astype(str)
        desc_entities.rename("name", inplace=True)
        desc_entities = desc_entities.to_frame()

        # Adopt the probability of the entity to which the descriptor is attached
        desc_entities["prob"] = entities["prob"].loc[desc_entities.index]

        desc_entities["source"] = "desc_create_entities - " + entities["name"].loc[desc_entities.index]

        return self.incorporate_new_entities(entities, desc_entities)

    def _parse_uncertainty_descriptors(self, entities: pd.DataFrame, descriptors: pd.DataFrame):
        """
        Extract entity probabilities from uncertainty descriptors.

        Adopts the lowest probability if an entity has multiple uncertainty descriptors.
        """
        if self.uncertainty_map is None:
            entities["prob"] = 1.0
            return entities

        print("Applying uncertainty...")
        prob = descriptors["name"].map(
            self.uncertainty_map,
        ).dropna().groupby(level=[0, 1, 2]).min().rename("prob")

        entities = entities.join(prob, how="left")
        entities["prob"] = entities["prob"].fillna(1.0)

        return entities

    def _extract_super_entities(self, entities: pd.DataFrame):
        """
        Extract super entities and drop entities requiring descriptors which have none.

        Remove those from entities which require descriptors because they don't provide
        enough information alone, e.g., "Rhythm"

        Propagate uncertainty up through super entities, but become more certain when
        propagating from the specific to the general. E.g., third the
        uncertainty for each propagation (get 1/3rd closer to 100)
        """
        def extract_super_entities(entities: pd.DataFrame, sup: bool = False):
            if len(entities) == 0:
                return []

            if sup:
                combine = [entities.copy()]
            else:
                combine = []

            # Get super entities
            entities_sup = entities.reset_index().merge(
                self.sup_map,
                on="name",
            ).set_index(["sample", "pattern", "entity"])
            entities_sup.drop("name", axis=1, inplace=True)
            entities_sup.rename({"sup": "name"}, axis=1, inplace=True)

            # Increase probability as the entities presumably become less specific,
            # and therefore more likely
            entities_sup["prob"] = \
                entities_sup["prob"] + (1 - entities_sup["prob"]) * 0.3

            combine.extend(extract_super_entities(entities_sup, sup=True))

            return combine

        print("Recursively extract super entities...")

        entities_sup = entities[["name", "prob"]].copy()
        entities_sup["source"] = "sup - " + entities["name"]
        entities_sups = extract_super_entities(entities_sup)

        if len(entities_sups) == 0:
            return entities

        # Combine the entities with the extracted super entities
        entities_sup = pd.concat(
            entities_sups
        )
        entities = self.incorporate_new_entities(entities, entities_sup)

        return entities

    def _parse_traveling_descriptors(self, entities, descriptors):
        print("Parsing traveling descriptors...")
        for traveling_desc_tem in self.traveling_descriptor_templates:
            subgraph = self.ancestor_subgraph(
                traveling_desc_tem.entity,
                include_root=True,
            )
            nodes = pd.Series(list(subgraph.nodes))
            entities_in_subgraph = entities[entities["name"].isin(nodes)]

            # Determine the descriptors which may be propagated
            desc_to_propagate = descriptors["name"][descriptors.index.isin(entities_in_subgraph.index)]
            desc_to_propagate = desc_to_propagate[
                desc_to_propagate.isin(traveling_desc_tem.descriptors)
            ].rename("name").to_frame()
            desc_to_propagate["prop_from"] = entities["name"].loc[desc_to_propagate.index]

            # Join with other entities coming from the same pattern
            desc_to_propagate["source"] = " [" + desc_to_propagate.index.map(str) + "]"
            desc_to_propagate = desc_to_propagate.reset_index(level=2).rename({"entity": "entity_from"}, axis=1)
            propagate_pairs = desc_to_propagate.join(
                entities_in_subgraph.rename({"name": "prop_to"}, axis=1).reset_index(
                    level=2
                ).rename({"entity": "entity_to"}, axis=1)[["prop_to", "entity_to"]],
                how="inner",
            )

            # Filter pairs to those which are proper ancestors
            propagate_pairs["prop_from_idx"] = propagate_pairs["prop_from"].map(self.entity_idx_map)
            propagate_pairs["prop_to_idx"] = propagate_pairs["prop_to"].map(self.entity_idx_map)
            propagate_pairs = propagate_pairs[
                self.ancestor_matrix[
                    propagate_pairs["prop_from_idx"].values,
                    propagate_pairs["prop_to_idx"].values,
                ]
            ]

            # Create the final descriptors to incorporate
            propagate_descriptors = propagate_pairs.rename({"entity_to": "entity"}, axis=1).set_index("entity", append=True)
            propagate_descriptors["source"] = "propagated - " + propagate_descriptors["prop_from"] + propagate_descriptors["source"]
            propagate_descriptors = propagate_descriptors[["name", "source"]]

            descriptors = pd.concat([descriptors, propagate_descriptors]).sort_index()

        return descriptors

    def _parse_attached_descriptors(self, entities, descriptors):
        print("Parsing attached descriptors...")
        entities_add = []
        for attached_desc_tem in self.attached_descriptor_templates:
            entity_subset = entities[entities["name"] == attached_desc_tem.entity]
            descriptor_subset = descriptors[descriptors.index.isin(entity_subset.index)]
            descriptor_subset = descriptor_subset[
                descriptor_subset["name"] == attached_desc_tem.descriptor
            ]
            descriptor_subset["name"] = attached_desc_tem.resulting_entity
            descriptor_subset["prob"] = entity_subset["prob"].loc[descriptor_subset.index]

            descriptor_subset["source"] = "attached - " + attached_desc_tem.entity + " - " + attached_desc_tem.descriptor
            entities_add.append(descriptor_subset)

        if len(entities_add) == 0:
            return entities

        return self.incorporate_new_entities(
            entities,
            pd.concat(entities_add),
            extract_super=True,
        )

    def _extract_compound_entities(self, entities, descriptors):
        """
        NOTE: We have to get the super before we can parse the compound patterns - which we'll then have to run through super themselves

        For example:
        Bifascicular block = RBBB with fascicular block (LAFB or LPFB)

        If we have "RBBB and LAFB" labeled, but it requires "Fascicular block" and we don't parse the super first, then we'll miss it...
        """
        def find_matching_entities(entity, entities):
            matching_entities = entities[entities["name"] == str(entity)]

            if len(entity.descriptors) == 0:
                # If no required descriptors, consider the samples with matching entities
                return matching_entities.index.get_level_values(0).drop_duplicates()

            # Otherwise, determine the required descriptors
            required_desc = pd.Series(entity.descriptors).explode().astype(str)

            # Get the descriptors of the matching entities
            matching_entity_desc = descriptors[descriptors.index.isin(matching_entities.index)]

            # Consider only the matching entities with all required descriptors
            desc_match = matching_entity_desc.groupby(level=[0, 1]).apply(
                lambda group: required_desc.isin(group["name"]).all()
            )

            desc_match = desc_match.index[desc_match]
            if len(desc_match) == 0:
                # Necessary since `get_level_values` raising an error when the index is empty
                return pd.Index([], dtype='int64', name='sample')

            return desc_match.get_level_values(0).drop_duplicates()

        # Incorporate compound entities
        print("Creating compound entities...")
        compound_entities = []
        for compound_tem in self.compound_templates:
            compound_tem.strings_to_instances(self.entity_template_map, self.descriptor_template_map)

            # Find samples to include
            include_matches = pd.concat([
                find_matching_entities(entity, entities).to_series() \
                    for entity in compound_tem.including
            ])
            all_included = include_matches.value_counts() == len(compound_tem.including)
            included_idx = all_included.index[all_included]

            # Stop here if no satisfying entities
            if len(included_idx) == 0:
                compound_entities.append(pd.DataFrame(columns=["name", "prob", "source"]))
                continue

            # Find samples which must be excluded
            if len(compound_tem.excluding) > 0:
                # Exclude any samples if they have any of the "excluding" entities
                exclude_matches = pd.concat([
                    find_matching_entities(entity, entities).to_series() \
                        for entity in compound_tem.excluding
                ])
                included_idx = included_idx[~included_idx.isin(exclude_matches.index)]

            # Create the compound entities
            added_compound_entities = pd.DataFrame([], index=included_idx)
            added_compound_entities["name"] = str(compound_tem.entity)
            added_compound_entities["prob"] = 1.0
            added_compound_entities["source"] = "compound - " + str(compound_tem)

            compound_entities.append(added_compound_entities)

        if len([
            added for added in compound_entities if len(added) > 0
        ]) == 0:
            return entities

        compound_entities = pd.concat([
            added for added in compound_entities if len(added) > 0
        ])
        compound_entities.index.names = ["sample"]
        print(f"Created {len(compound_entities)} compound entities.")

        return self.incorporate_new_entities(
            entities,
            compound_entities,
            extract_super=True,
        )

    def _manipulate_entities(self, entities, descriptors):
        # Mark which entities/descriptors come directly from the text before
        # incorporating more
        entities["source"] = "text"
        descriptors["source"] = "text"

        # Compute uncertainty from uncertainty descriptors
        entities = self._parse_uncertainty_descriptors(entities, descriptors)

        # Check for descriptors without templates
        desc_mapped = descriptors["name"].map(self.descriptor_template_map)

        if desc_mapped.isna().any():
            raise ValueError(
                'Missing descriptor templates for: '
                f'{list(descriptors["name"][desc_mapped.isna()].unique())}'
            )

        # desc_mapped = descriptors["name"].map(self.descriptor_template_map)
        # descriptors = descriptors[~desc_mapped.isna()]

        # Extract descriptor `create_entities` entities
        entities = self._extract_desc_entities(entities, descriptors)

        # Ceil certain entities
        entities.loc[entities["name"].isin(["Needs clinical correlation"]), "prob"] = 1.0

        # Drop any entities with the "Absence" descriptor
        len_before = len(entities)
        absent = descriptors["name"] == "Absence"
        entities[~entities.index.isin(descriptors.index[absent])]
        descriptors = descriptors[~absent]
        print(f"Dropped {len_before - len(entities)} entities with absence descriptors.")

        # Extract super entities
        entities = self._extract_super_entities(entities)

        # Parse traveling descriptors
        descriptors = self._parse_traveling_descriptors(entities, descriptors)

        # Parse attached descriptors
        entities = self._parse_attached_descriptors(entities, descriptors)

        # Extract compound entities
        entities = self._extract_compound_entities(entities, descriptors)

        # Note: May be tempted to drop duplicates here
        # I.e., (rows with the same name, source, and prob)
        # But we can't because they are not guarenteed to have the same descriptors

        return entities, descriptors

    def _extract_labels(
        self,
        entities: pd.DataFrame,
        descriptors: pd.DataFrame,
    ):
        """
        Create labels.
        """
        # Create descriptor labels
        descriptor_labels = descriptors.join(
            entities["name"].rename("entity"),
            how="left",
        )
        descriptor_labels["name"] = descriptor_labels["entity"] + " - " + \
            descriptor_labels["name"].astype(str)
        descriptor_labels["prob"] = 1.0
        descriptor_labels = descriptor_labels[["name", "prob"]].copy()

        # Drop if requries descriptors - we've extracted the "Entity - Descriptor"
        # labels already, so we can only remove these entities
        requires_descriptors = entities["name"].map(self.entity_template_map).apply(lambda tem: tem.require_descriptors)
        entity_labels = entities[~requires_descriptors][["name", "prob"]].copy()

        # Create labels
        entity_labels["label_type"] = "entity"
        descriptor_labels["label_type"] = "descriptor"

        labels = pd.concat([
            entity_labels,
            descriptor_labels,
        ])
        labels.sort_index(inplace=True)

        labels_flat = labels.reset_index().groupby(
            ["sample", "name"]
        ).agg({"prob": "max"}).reset_index(level=1)

        return labels, labels_flat

    @pickle_on_error_with_provided_path(
        attribute="res",
        restore_path_arg="restore_path",
        pickle_on_return=True,
    )
    def __call__(
        self,
        texts: Optional[pd.Series] = None,
        res: Optional[PatternLabelerResults] = None,
        restore_path: Optional[str] = None,
    ):
        """
        Store results within the object to make debugging more convenient if an error is raised
        Can obtain/save partial results from the PatternLabeler, make certain fixes,
        then continue on the same step providing the object with `res=`
        """
        # Load from provided PatternLabelerResults object
        if res is not None:
            assert texts is None
            assert isinstance(res, PatternLabelerResults)
            assert res.texts is not None
            texts = res.texts.copy()
        # Load from provided file if it exists
        elif restore_path is not None and os.path.isfile(restore_path):
            with open(restore_path, "rb") as f:
                res = pickle.load(f)

            if texts is not None:
                raise ValueError(
                    "Pulling texts from loaded labeler. Do not specify `texts` kwarg."
                )

            texts = res.texts.copy()
        # Begin processing from texts
        elif texts is not None:
            texts = texts.copy()
            res = PatternLabelerResults(texts=texts)
        else:
            raise ValueError(
                "Must specify one of `texts`, `res`, `restore_file`. "
                "If `restore_file` alone is specified, the file must exist."
            )

        self.res = res

        # Match patterns
        if res.text_results is None or res.pattern_results is None:
            res.text_results, res.pattern_results = self._match(texts)

        # Create entities and descriptors
        if res.pattern_results is None or res.entities_text is None:
            res.pattern_results, res.entities_text, res.descriptors_text = \
                self._extract_entities(res.pattern_results)

        # Manipulate entities and descroptors
        if res.entities is None or res.descriptors is None:
            entities = res.entities_text.copy()
            descriptors = res.descriptors_text.copy()
            res.entities, res.descriptors = \
                self._manipulate_entities(entities, descriptors)

        # Create labels
        if res.labels is None or res.labels_flat is None:
            res.labels, res.labels_flat = \
                self._extract_labels(res.entities.copy(), res.descriptors.copy())

        return res

    # UTILITY METHODS
    def label_to_matches(self, label: str):
        return self.res.texts[
            self.res.labels_per_entity[
                self.res.labels_per_entity["name"] == label
            ].index.get_level_values(0)
        ]

    def super_entity_percent(self, entities: pd.DataFrame, entities_sup: pd.DataFrame):
        """
        Useful for ... entity appears in a pattern versus is added as a super entity.
        """
        counts = entities["name"].value_counts().rename("", inplace=True).to_frame().join(
            entities_sup["name"].value_counts().rename("", inplace=True),
            lsuffix='ent_count',
            rsuffix='sup_count',
        )
        counts = counts.fillna(0)
        counts["sup_count"] = counts["sup_count"].astype(int)
        counts["diff"] = counts["sup_count"] - counts["ent_count"]

        # Ignore removed entities, e.g., since they didn't have a descriptor
        counts["diff"][counts["diff"] < 0] = 0

        # Compute percent super entities versus matched entities
        counts["percent_sup"] = (counts["sup_count"] - counts["ent_count"]) / \
            (counts["ent_count"] + counts["sup_count"])

        counts.sort_values("percent_sup", ascending=False, inplace=True)

        return counts

    def ancestor_subgraph(self, entity: str, include_root: bool = True):
        assert isinstance(entity, str)
        ancestors = nx.ancestors(self.sup_graph, entity)

        if include_root:
            ancestors = {entity}.union(ancestors)

        return self.sup_graph.subgraph(ancestors).copy()

    def plot_ancestor_subgraph(self, entity: str, figsize=(15, 8), **kwargs):
        subgraph = self.ancestor_subgraph(entity, include_root=True)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        nx.draw_networkx(subgraph, with_labels=True, ax=ax, **kwargs)

    def cull(self):
        if self.res is None:
            raise ValueError(
                "The pattern labeler must be run before determining how to cull."
            )

        # Cull templates
        ent_tem_used = self.res.entities['name'].unique()
        self.entity_templates = [
            tem for tem in self.entity_templates if tem.name in ent_tem_used
        ]

        # Handling matched descriptor patterns
        # Some matched descriptors never get attributed, but
        # they still need to be accounted for
        desc_matched = self.res.pattern_results[
            self.res.pattern_results['type'].isin(['b', 'a'])
        ]['pattern'].apply(lambda pat: pat.descriptors).explode().dropna()

        desc_tem = desc_matched[
            desc_matched.apply(type).apply(lambda class_type: class_type.__name__) == DescriptorTem.__name__
        ]
        desc = desc_matched[
            desc_matched.apply(type).apply(lambda class_type: class_type.__name__) == Descriptor.__name__
        ]
        desc_desc = desc.apply(lambda desc: desc.descriptors).explode().dropna().astype(str)

        desc_tem_all = pd.concat([desc_tem, desc.apply(lambda desc: desc.tem)]).astype(str)
        desc_tem_entities = desc_tem_all.map(
            self.descriptor_template_map
        ).apply(lambda tem: tem.create_entities).explode().dropna()
        desc_tem_used = pd.concat([
            desc_tem_all, # All matched
            desc_desc, # DescriptorTem's `descriptors`
            pd.Series(self.res.descriptors['name']),
        ]).unique()
        self.descriptor_templates = [
            tem for tem in self.descriptor_templates if tem.name in desc_tem_used
        ]

        # Cull patterns
        pat_used = self.res.pattern_results['pattern_str'].unique()

        self.entity_patterns = [
            pat for pat in self.entity_patterns if pat.pattern in pat_used
        ]

        self.descriptor_patterns = [
            pat for pat in self.descriptor_patterns if pat.pattern in pat_used
        ]

        self.connectives = [
            pat for pat in self.connectives if pat.pattern in pat_used
        ]

        # Cull relational based on available templates/patterns
        def valid_descriptor(descriptor):
            if isinstance(descriptor, str):
                return descriptor in desc_tem_used

            if isinstance(descriptor, DescriptorTem):
                return entity.name in desc_tem_used

            if isinstance(descriptor, Descriptor):
                return str(descriptor.tem) in desc_tem_used and \
                    all(valid_descriptor(desc) for desc in descriptor.descriptors)

            raise ValueError("Unrecognized descriptor type.")

        def valid_entity(entity):
            if isinstance(entity, str):
                return entity in ent_tem_used

            if isinstance(entity, EntityTem):
                return entity.name in ent_tem_used

            if isinstance(entity, Entity):
                return str(entity.tem) in ent_tem_used and \
                    all(valid_descriptor(desc) for desc in entity.descriptors)

            raise ValueError("Unrecognized entity type.")

        def valid_compound(compound_template):
            if not valid_entity(compound_template.entity):
                return False

            for i, entity in enumerate(compound_template.including):
                if not valid_entity(entity):
                    return False

            for i, entity in enumerate(compound_template.excluding):
                if not valid_entity(entity):
                    return False

            return True

        def valid_attached_descriptor(attached_descriptor_template):
            return valid_entity(attached_descriptor_template.resulting_entity) and \
            valid_entity(attached_descriptor_template.entity) and \
            valid_descriptor(attached_descriptor_template.descriptor)

        def modify_traveling_descriptors(traveling_descriptor_templates):
            traveling_keep = []
            for i, tem in enumerate(traveling_descriptor_templates):
                if not valid_entity(tem.entity):
                    continue

                desc_keep = []
                for desc in tem.descriptors:
                    if valid_descriptor(desc):
                        desc_keep.append(desc)

                if len(desc_keep) == 0:
                    continue

                traveling_keep.append(TravelingDescriptorsTem(tem.entity, desc_keep))

            return traveling_keep

        self.compound_templates = [
            tem for tem in self.compound_templates if valid_compound(tem)
        ]

        self.attached_descriptor_templates = [
            tem for tem in self.attached_descriptor_templates if valid_attached_descriptor(tem)
        ]

        self.traveling_descriptor_templates = modify_traveling_descriptors(
            self.traveling_descriptor_templates
        )

        self.uncertainty_map = {
            key: val for key, val in self.uncertainty_map.items() if key in desc_tem_used
        }

        # Return an updated labeler config
        # This makes it easy to save the culled version using the .to_json method
        descriptor_templates = [
            tem for tem in self.descriptor_templates if isinstance(tem, DescriptorTem)
        ]
        split_descriptor_templates = [
            tem for tem in self.descriptor_templates if isinstance(tem, SplitDescriptorTem)
        ]

        labeler_config = PatternLabelerConfig(
            self.entity_templates,
            self.entity_patterns,
            descriptor_templates=descriptor_templates,
            descriptor_patterns=self.descriptor_patterns,
            split_descriptor_templates=split_descriptor_templates,
            connectives=self.connectives,
            compound_templates=self.compound_templates,
            attached_descriptor_templates=self.attached_descriptor_templates,
            traveling_descriptor_templates=self.traveling_descriptor_templates,
            uncertainty_map=self.uncertainty_map,
        )

        return labeler_config
