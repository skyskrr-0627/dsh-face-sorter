"""Convert the usage guide Markdown into a shareable DOCX (subset of Markdown)."""
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

BODY_FONT = "PingFang SC"
CODE_FONT = "Menlo"
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
MUTED = RGBColor(0x5A, 0x61, 0x6B)


def set_cjk(run, font=BODY_FONT):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), font)


def shade(paragraph_or_cell, hex_fill):
    pr = paragraph_or_cell._element.get_or_add_pPr() if hasattr(paragraph_or_cell._element, "get_or_add_pPr") else None
    if pr is None:
        return
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    pr.append(shd)


INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|<https?://[^>]+>)")


def add_inline(paragraph, text, base_size=10.5, base_color=None, bold=False):
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
            set_cjk(run)
            run.font.size = Pt(base_size)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            run = paragraph.add_run(piece[1:-1])
            set_cjk(run, CODE_FONT)
            run.font.size = Pt(base_size - 0.5)
            run.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
        elif piece.startswith("<http") and piece.endswith(">"):
            run = paragraph.add_run(piece[1:-1])
            set_cjk(run)
            run.font.size = Pt(base_size)
            run.font.color.rgb = ACCENT
        else:
            run = paragraph.add_run(piece)
            set_cjk(run)
            run.font.size = Pt(base_size)
            run.bold = bold
        if base_color is not None and not piece.startswith("`") and not piece.startswith("<http"):
            run.font.color.rgb = base_color


def build(md_path, out_path):
    lines = open(md_path, encoding="utf-8").read().split("\n")
    doc = Document()
    section = doc.sections[0]
    from docx.shared import Cm

    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    next_subtitle = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip())
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.4)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            shade(p, "F2F4F7")
            for n, text in enumerate(block):
                run = p.add_run(text)
                set_cjk(run, CODE_FONT)
                run.font.size = Pt(9)
                if n < len(block) - 1:
                    run.add_break()
            continue

        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                width = len(rows[0])
                table = doc.add_table(rows=0, cols=width)
                table.style = "Light Grid Accent 1"
                table.alignment = WD_TABLE_ALIGNMENT.LEFT
                for r_idx, cells in enumerate(rows):
                    cells = (cells + [""] * width)[:width]
                    row = table.add_row()
                    for c_idx, text in enumerate(cells):
                        cell = row.cells[c_idx]
                        cell.text = ""
                        para = cell.paragraphs[0]
                        add_inline(para, text, base_size=9.5, bold=(r_idx == 0))
                        for run in para.runs:
                            run.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
                        if r_idx == 0:
                            shade(para, "E8EEF9")
                doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:
                para = doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.paragraph_format.space_after = Pt(6)
                add_inline(para, text, base_size=20, bold=True)
                next_subtitle = True
            else:
                heading = doc.add_heading(level=level - 1)
                heading.paragraph_format.space_before = Pt(14 if level == 2 else 10)
                heading.paragraph_format.space_after = Pt(4)
                for run in list(heading.runs):
                    run.text = ""
                add_inline(heading, text, base_size=15 if level == 2 else 12, bold=True)
                for run in heading.runs:
                    run.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
            i += 1
            continue

        if stripped in ("---", "***"):
            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after = Pt(2)
            ppr = para._p.get_or_add_pPr()
            borders = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "D5DAE2")
            borders.append(bottom)
            ppr.append(borders)
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            para = doc.add_paragraph()
            para.paragraph_format.space_after = Pt(3)
            para.paragraph_format.left_indent = Cm(0.85)
            para.paragraph_format.first_line_indent = Cm(-0.85)
            add_inline(para, f"{m.group(1)}. {m.group(2)}")
            i += 1
            continue

        if stripped.startswith("- "):
            para = doc.add_paragraph(style="List Bullet")
            para.paragraph_format.space_after = Pt(3)
            add_inline(para, stripped[2:])
            i += 1
            continue

        if next_subtitle:
            next_subtitle = False
            sub = doc.add_paragraph()
            sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
            sub.paragraph_format.space_after = Pt(16)
            add_inline(sub, stripped, base_size=10, base_color=MUTED)
            i += 1
            continue

        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.line_spacing = 1.25
        add_inline(para, stripped)
        i += 1

    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    print(build(sys.argv[1], sys.argv[2]))
