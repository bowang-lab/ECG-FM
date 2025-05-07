from typing import List
import os
import json

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

def json_to_entity_tem(directory: str) -> List[EntityTem]:
    with open(os.path.join(directory, 'entity_tem.json'), 'r') as f:
        data = json.load(f)
    
    return [EntityTem(**item) for item in data]

def json_to_entity_pat(directory: str) -> List[EntityPattern]:
    def form(pat_data):
        pat_data['entities'] = to_list(pat_data['entities'])
        for i, entity in enumerate(pat_data['entities']):
            if isinstance(entity, dict):
                pat_data['entities'][i] = Entity(**pat_data['entities'][i])

        return EntityPattern(**pat_data)

    with open(os.path.join(directory, 'entity_pat.json'), 'r') as f:
        data = json.load(f)

    return [form(pat_data) for pat_data in data]

def json_to_desc_tem(directory: str) -> List[DescriptorTem]:
    with open(os.path.join(directory, 'desc_tem.json'), 'r') as f:
        data = json.load(f)

    return [DescriptorTem(**item) for item in data]

def json_to_desc_pat(directory: str) -> List[DescriptorPattern]:
    with open(os.path.join(directory, 'desc_pat.json'), 'r') as f:
        data = json.load(f)

    def form(pat_data):
        pat_data['descriptors'] = to_list(pat_data['descriptors'])
        for i, desc in enumerate(pat_data['descriptors']):
            if isinstance(desc, dict):
                pat_data['descriptors'][i] = Descriptor(**pat_data['descriptors'][i])

        return DescriptorPattern(**pat_data)

    with open(os.path.join(directory, 'desc_pat.json'), 'r') as f:
        data = json.load(f)

    return [form(pat_data) for pat_data in data]

def json_to_split_desc_tem(directory: str) -> List[SplitDescriptorTem]:
    with open(os.path.join(directory, 'split_desc_tem.json'), 'r') as f:
        data = json.load(f)

    return [SplitDescriptorTem(**item) for item in data]

def json_to_connective(directory: str) -> List[Connective]:
    with open(os.path.join(directory, 'connective.json'), 'r') as f:
        data = json.load(f)

    return [Connective(**item) for item in data]

def json_to_compound_tem(directory: str) -> List[CompoundTem]:
    with open(os.path.join(directory, 'compound_tem.json'), 'r') as f:
        data = json.load(f)

    def form(tem_data):
        tem_data['including'] = [
            Entity(**item) if isinstance(item, dict) else item \
                for item in tem_data['including']
        ]
        tem_data['excluding'] = [
            Entity(**item) if isinstance(item, dict) else item \
                for item in tem_data['excluding']
        ]

        return CompoundTem(**tem_data)

    return [form(tem_data) for tem_data in data]

def json_to_attached_desc_tem(directory: str) -> List[AttachedDescriptorTem]:
    with open(os.path.join(directory, 'attached_desc_tem.json'), 'r') as f:
        data = json.load(f)

    return [AttachedDescriptorTem(**item) for item in data]

def json_to_traveling_desc_tem(directory: str) -> List[TravelingDescriptorsTem]:
    with open(os.path.join(directory, 'traveling_desc_tem.json'), 'r') as f:
        data = json.load(f)

    return [TravelingDescriptorsTem(**item) for item in data]

def json_to_uncertainty_map(directory: str) -> List[TravelingDescriptorsTem]:
    with open(os.path.join(directory, 'uncertainty_map.json'), 'r') as f:
        data = json.load(f)

    return data

FROM_JSON_FUNC = {
    'entity_tem': json_to_entity_tem,
    'entity_pat': json_to_entity_pat,
    'desc_tem': json_to_desc_tem,
    'desc_pat': json_to_desc_pat,
    'split_desc_tem': json_to_split_desc_tem,
    'connective': json_to_connective,
    'compound_tem': json_to_compound_tem,
    'attached_desc_tem': json_to_attached_desc_tem,
    'traveling_desc_tem': json_to_traveling_desc_tem,
    'uncertainty_map': json_to_uncertainty_map,
}
