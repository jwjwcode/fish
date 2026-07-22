"""Minimal Excel workbook writer for compact score outputs."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


SCORE_COLUMNS = [
    "time_s",
    "current_segmentation_score",
    "current_optical_flow_score",
    "current_total_activity",
    "previous_10_total_activity_average",
]


def column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def cell_xml(row_index: int, column_index: int, value: Any) -> str:
    ref = f"{column_letter(column_index)}{row_index}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, Real) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{float(value):.10g}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def worksheet_xml(rows: Iterable[Iterable[Any]]) -> str:
    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = [
            cell_xml(row_index, column_index, value)
            for column_index, value in enumerate(row, start=1)
        ]
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        "</worksheet>"
    )


def write_score_workbook(
    path: Path,
    score_rows: list[dict[str, float]],
    final_feeding_score: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_sheet_rows: list[list[Any]] = [SCORE_COLUMNS]
    for row in score_rows:
        score_sheet_rows.append([row.get(column) for column in SCORE_COLUMNS])

    summary_rows = [
        ["metric", "value"],
        ["final_feeding_score", final_feeding_score],
        ["processed_frame_count", len(score_rows)],
    ]

    with ZipFile(path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        workbook.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            '<sheet name="scores" sheetId="1" r:id="rId1"/>'
            '<sheet name="summary" sheetId="2" r:id="rId2"/>'
            "</sheets>"
            "</workbook>",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
            "</Relationships>",
        )
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet_xml(score_sheet_rows))
        workbook.writestr("xl/worksheets/sheet2.xml", worksheet_xml(summary_rows))
