# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
import pandas as pd


class TABLE_TOKEN(Enum):
    TAB = "[TAB]"
    ROW = "[ROW]"
    CELL = "[CELL]"


def prompt_tuning_table_prompt(table: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a string with paper's special tokens.

    Format (per paper Eq. 1 / Appendix F):
        [TAB] [ROW] [CELL] col1 [CELL] col2 [ROW] [CELL] val1 [CELL] val2

    [TAB] marks the start of the table, [ROW] marks each row (including the
    header row), and [CELL] is a prefix before every cell value.
    """
    parts = [TABLE_TOKEN.TAB.value]
    # Header row
    parts.append(TABLE_TOKEN.ROW.value)
    for col in table.columns:
        parts.append(TABLE_TOKEN.CELL.value)
        parts.append(str(col))
    # Data rows
    for _, row in table.iterrows():
        parts.append(TABLE_TOKEN.ROW.value)
        for val in row.values:
            parts.append(TABLE_TOKEN.CELL.value)
            parts.append(str(val))
    return " ".join(parts)
