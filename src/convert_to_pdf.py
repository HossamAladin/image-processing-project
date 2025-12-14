#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert BEGINNER_GUIDE.md to PDF format."""
import os
import sys
import markdown
from xhtml2pdf import pisa

md_file = "BEGINNER_GUIDE.md"
pdf_file = "BEGINNER_GUIDE.pdf"

if not os.path.exists(md_file):
    print(f"Error: {md_file} not found in current directory!")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)

print(f"Reading {md_file}...")
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

print("Converting markdown to HTML...")
html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite', 'tables'])

css_style = """<style>
@page { size: A4; margin: 2cm; }
body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 30px; }
h3 { color: #555; margin-top: 20px; }
code { background-color: #f4f4f4; padding: 2px 6px; font-family: monospace; }
pre { background-color: #f4f4f4; padding: 15px; border-left: 4px solid #3498db; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
th { background-color: #3498db; color: white; }
ul, ol { margin: 10px 0; padding-left: 30px; }
li { margin: 5px 0; }
</style>"""

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Image Processing Project - Beginner's Guide</title>
{css_style}
</head>
<body>
{html_content}
</body>
</html>"""

print(f"Generating PDF: {pdf_file}...")
try:
    with open(pdf_file, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(full_html, dest=result_file, encoding='utf-8')
    
    if pisa_status.err:
        print(f"Error creating PDF: {pisa_status.err}")
        sys.exit(1)
    else:
        print(f"Success! PDF created: {pdf_file}")
        if os.path.exists(pdf_file):
            size_kb = os.path.getsize(pdf_file) / 1024
            print(f"File size: {size_kb:.1f} KB")
except Exception as e:
    print(f"Error creating PDF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
