#!/usr/bin/env python
"""Convert BEGINNER_GUIDE.md to PDF format."""
import os
import markdown
from weasyprint import HTML
from weasyprint.text.fonts import FontConfiguration

# Read the markdown file
# Check both current directory and src directory
if os.path.exists("BEGINNER_GUIDE.md"):
    md_file = "BEGINNER_GUIDE.md"
    pdf_file = "BEGINNER_GUIDE.pdf"
elif os.path.exists("src/BEGINNER_GUIDE.md"):
    md_file = "src/BEGINNER_GUIDE.md"
    pdf_file = "BEGINNER_GUIDE.pdf"
else:
    md_file = None
    pdf_file = "BEGINNER_GUIDE.pdf"

if md_file is None or not os.path.exists(md_file):
    print(f"Error: BEGINNER_GUIDE.md not found!")
    print(f"Current directory: {os.getcwd()}")
    print("Please make sure BEGINNER_GUIDE.md exists in current directory or src/")
    exit(1)

print(f"Reading {md_file}...")
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
print("Converting markdown to HTML...")
html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite', 'tables'])

# Add CSS styling for better PDF appearance
css_style = """
<style>
    @page {
        size: A4;
        margin: 2cm;
    }
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 100%;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        page-break-after: avoid;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #95a5a6;
        padding-bottom: 5px;
        margin-top: 30px;
        page-break-after: avoid;
    }
    h3 {
        color: #555;
        margin-top: 20px;
        page-break-after: avoid;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9em;
    }
    pre {
        background-color: #f4f4f4;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    th {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    blockquote {
        border-left: 4px solid #3498db;
        margin: 15px 0;
        padding-left: 15px;
        color: #666;
        font-style: italic;
    }
    ul, ol {
        margin: 10px 0;
        padding-left: 30px;
    }
    li {
        margin: 5px 0;
    }
    strong {
        color: #2c3e50;
    }
    a {
        color: #3498db;
        text-decoration: none;
    }
    hr {
        border: none;
        border-top: 2px solid #ecf0f1;
        margin: 30px 0;
    }
</style>
"""

# Combine HTML
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Image Processing Project - Beginner's Guide</title>
    {css_style}
</head>
<body>
    {html_content}
</body>
</html>
"""

# Convert HTML to PDF
print(f"Generating PDF: {pdf_file}...")
try:
    font_config = FontConfiguration()
    HTML(string=full_html).write_pdf(
        pdf_file,
        font_config=font_config
    )
    print(f"✅ Success! PDF created: {pdf_file}")
    print(f"   File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
except Exception as e:
    print(f"❌ Error creating PDF: {e}")
    import traceback
    traceback.print_exc()

