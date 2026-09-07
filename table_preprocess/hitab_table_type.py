# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def table_deep(node):
    """Compute depth of a HiTab header tree node (raw JSON dict)."""
    children = node.get('children_dict', [])
    if not children:
        return 1
    return 1 + max(table_deep(child) for child in children)
