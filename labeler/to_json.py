from typing import List
import os
import json

import pandas as pd

from common import to_list
from pattern_labeler import (
    AttachedDescriptorTem,
    CompoundTem,
    Connective,
    Descriptor,
    DescriptorPattern,
    DescriptorTem,
    Entity,
    EntityPattern,
    EntityTem,
    SplitDescriptorTem,
    TravelingDescriptorsTem,
)

def stringify(item):
    if isinstance(item, str):
        return item

    if item is None:
        return item

    if isinstance(item, list):
        return [str(sub_item) for sub_item in item]

    return str(item)

def entity_tem_to_json(directory: str, entity_templates: List[EntityTem]):
    def to_dict(tem):
        data = {
            "name": tem.name,
            "sup": stringify(tem.sup),
            "aka": tem.aka,
            "require_descriptors": tem.require_descriptors,
            "tags": tem.tags,
            **{k: v for k, v in tem.__dict__.items() if k not in [
                "name",
                "sup",
                "aka",
                "descriptors",
                "require_descriptors",
                "tags",
            ]}
        }

        return {key: val for key, val in data.items() if val != []}

    entity_tem_data = pd.Series(entity_templates).apply(to_dict)
    with open(os.path.join(directory, 'entity_tem.json'), 'w') as f:
        json.dump(entity_tem_data.values.tolist(), f, indent=2)

def entity_pat_to_json(directory, entity_patterns: List[EntityPattern]):
    def parse_entities(entities):
        if not isinstance(entities, list):
            entities = [entities]

        for i, entity in enumerate(entities):
            if isinstance(entity, EntityTem):
                entities[i] = {
                    'tem': entity.name,
                    'descriptors': stringify(entity.descriptors),
                }
            elif isinstance(entity, Entity):
                entities[i] = {
                    'tem': stringify(entity.tem),
                    'descriptors': stringify(entity.descriptors),
                }

        return entities

    def to_dict(pat):
        return {
            "pattern": pat.pattern,
            "entities": parse_entities(pat.entities),
        }

    entity_pat = pd.Series(entity_patterns).apply(to_dict)

    with open(os.path.join(directory, 'entity_pat.json'), 'w') as f:
        json.dump(entity_pat.values.tolist(), f, indent=2)

def desc_tem_to_json(directory, descriptor_templates: List[DescriptorTem]):
    def to_dict(tem):
        data = {
            "name": tem.name,
            "category": tem.category,
            "tags": tem.tags,
            "create_entities": stringify(tem.create_entities),
        }

        return {key: val for key, val in data.items() if val != []}

    descriptor_tem = pd.Series(descriptor_templates).apply(to_dict)
    with open(os.path.join(directory, 'desc_tem.json'), 'w') as f:
        json.dump(descriptor_tem.values.tolist(), f, indent=2)

def desc_pat_to_json(directory, descriptor_patterns: List[DescriptorPattern]):
    def parse_descriptors(descriptors):
        if not isinstance(descriptors, list):
            descriptors = [descriptors]

        for i, descriptor in enumerate(descriptors):
            if isinstance(descriptor, DescriptorTem):
                descriptors[i] = {
                    'tem': descriptor.name,
                    'descriptors': stringify(descriptor.descriptors),
                }

            elif isinstance(descriptor, Descriptor):
                descriptors[i] = {
                    'tem': stringify(descriptor.tem),
                    'descriptors': stringify(descriptor.descriptors),
                }

        return descriptors

    def to_dict(pat):
        """Convert EntityTem object to a dictionary."""
        return {
            "pattern": pat.pattern,
            "descriptors": parse_descriptors(pat.descriptors),
            "before_after": pat.before_after,
        }

    descriptor_pat = pd.Series(descriptor_patterns).apply(to_dict)
    with open(os.path.join(directory, 'desc_pat.json'), 'w') as f:
        json.dump(descriptor_pat.values.tolist(), f, indent=2)

def split_desc_tem_to_json(
    directory,
    split_descriptor_templates: List[SplitDescriptorTem],
):
    def to_dict(tem):
        data = {
            "name": tem.name, 
            "split": stringify(tem.split),
        }
        if tem.patterns is not None:
            data["patterns"] = tem.patterns

        return data

    split_descriptor_tem = pd.Series(split_descriptor_templates).apply(to_dict)
    with open(os.path.join(directory, 'split_desc_tem.json'), 'w') as f:
        json.dump(split_descriptor_tem.values.tolist(), f, indent=2)

def connective_to_json(directory, connectives: List[Connective]):
    def to_dict(connective):
        data = {
            "pattern": connective.pattern,
            "descriptors": [
                stringify(connective.descriptors[0]),
                stringify(connective.descriptors[1]),
            ],
            "tags": connective.tags,
        }

        return {key: val for key, val in data.items() if val != [] and val != None}

    connectives = pd.Series(connectives).apply(to_dict)
    with open(os.path.join(directory, 'connective.json'), 'w') as f:
        json.dump(connectives.values.tolist(), f, indent=2)

def compound_tem_to_json(directory, compound_templates: List[CompoundTem]):
    def to_dict(tem):
        data = {
            "entity": stringify(tem.entity),
            "including": tem.including,
            "excluding": tem.excluding,
        }

        for i, entity in enumerate(data["including"]):
            if isinstance(entity, Entity):
                data["including"][i] = {
                    'tem': stringify(entity.tem),
                    'descriptors': entity.descriptors,
                }

        for i, entity in enumerate(data["excluding"]):
            if isinstance(entity, Entity):
                data["excluding"][i] = {
                    'tem': stringify(entity.tem),
                    'descriptors': entity.descriptors,
                }

        return data

    compound_tem = pd.Series(compound_templates).apply(to_dict)
    with open(os.path.join(directory, 'compound_tem.json'), 'w') as f:
        json.dump(compound_tem.values.tolist(), f, indent=2)

def attached_desc_tem_to_json(
    directory,
    attached_descriptor_templates: List[AttachedDescriptorTem],
):
    def to_dict(tem):
        return {
            "resulting_entity": tem.resulting_entity, 
            "entity": tem.entity,
            "descriptor": tem.descriptor,
        }

    tem = pd.Series(attached_descriptor_templates).apply(to_dict)
    with open(os.path.join(directory, 'attached_desc_tem.json'), 'w') as f:
        json.dump(tem.values.tolist(), f, indent=2)

def traveling_desc_tem_to_json(
    directory,
    traveling_descriptor_templates: List[TravelingDescriptorsTem],
):
    def to_dict(tem):
        return {
            "entity": tem.entity, 
            "descriptors": to_list(tem.descriptors),
        }

    tem = pd.Series(traveling_descriptor_templates).apply(to_dict)
    with open(os.path.join(directory, 'traveling_desc_tem.json'), 'w') as f:
        json.dump(tem.values.tolist(), f, indent=2)

def uncertainty_map_to_json(directory, uncertainty_map: dict):
    with open(os.path.join(directory, 'uncertainty_map.json'), 'w') as f:
        json.dump(uncertainty_map, f, indent=2)

TO_JSON_FUNC = {
    'entity_tem': entity_tem_to_json,
    'entity_pat': entity_pat_to_json,
    'desc_tem': desc_tem_to_json,
    'desc_pat': desc_pat_to_json,
    'split_desc_tem': split_desc_tem_to_json,
    'connective': connective_to_json,
    'compound_tem': compound_tem_to_json,
    'attached_desc_tem': attached_desc_tem_to_json,
    'traveling_desc_tem': traveling_desc_tem_to_json,
    'uncertainty_map': uncertainty_map_to_json,
}