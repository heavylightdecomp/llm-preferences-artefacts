#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved script to fix malformed tab-separated CSV files.

This version properly handles:
1. Multi-line content within fields
2. Embedded tabs in text content
3. HTML entities
4. Character encoding issues
5. Proper field quoting
"""

import csv
import html
import re
import sys
from pathlib import Path

def fix_text_content(text):
    """Fix text content by handling embedded tabs and HTML entities."""
    if not text:
        return text
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Replace embedded tabs with spaces (preserve the TSV structure)
    text = text.replace('\t', ' ')
    
    # Replace multiple consecutive spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â€"', '—')
    text = text.replace('â€"', '–')
    
    # Replace newlines with spaces to keep everything on one line
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text.strip()

def fix_csv_file(input_file, output_file):
    """Fix a malformed TSV file using proper CSV parsing."""
    print(f"Processing {input_file}...")
    
    # Read the original file as raw text
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    if not lines:
        print("Empty file!")
        return
    
    # Process header
    header = lines[0].strip()
    print(f"Header: {header}")
    
    # Count expected columns based on header
    expected_columns = len(header.split('\t'))
    print(f"Expected columns: {expected_columns}")
    
    # Process data lines
    fixed_rows = []
    current_row = []
    current_field = ""
    in_quotes = False
    field_count = 0
    
    # Start from the second line (skip header)
    for line_num, line in enumerate(lines[1:], 1):
        line = line.strip()
        if not line:
            continue
        
        # Split by tabs
        parts = line.split('\t')
        
        # If this is the start of a new row (has the expected number of parts or fewer)
        if len(parts) <= expected_columns and field_count == 0:
            # This is a new row
            if current_row:
                # Save the previous row
                fixed_rows.append(current_row)
            
            # Start new row
            current_row = []
            field_count = 0
            
            # Process this line
            for part in parts:
                if field_count < expected_columns:
                    fixed_part = fix_text_content(part) if field_count >= 6 else part.replace('\t', ' ')
                    current_row.append(fixed_part)
                    field_count += 1
            
            # If we have a complete row, save it
            if field_count == expected_columns:
                fixed_rows.append(current_row)
                current_row = []
                field_count = 0
        else:
            # This is a continuation of the previous field (likely Article Text)
            if current_row and field_count < expected_columns:
                # Append to the current field
                current_row[-1] += ' ' + fix_text_content(line)
            elif current_row:
                # We're past the expected number of fields, append to the last field
                current_row[-1] += ' ' + fix_text_content(line)
    
    # Don't forget the last row
    if current_row:
        fixed_rows.append(current_row)
    
    # Write the fixed file using proper CSV writer
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        writer.writerow(header.split('\t'))
        
        # Write data rows
        for row in fixed_rows:
            # Ensure we have the right number of columns
            while len(row) < expected_columns:
                row.append('')
            row = row[:expected_columns]  # Truncate if too many
            writer.writerow(row)
    
    print(f"Fixed file saved as: {output_file}")
    print(f"Processed {len(fixed_rows)} data rows")

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_csv_v2.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Input file {input_file} does not exist!")
        sys.exit(1)
    
    fix_csv_file(input_file, output_file)

if __name__ == "__main__":
    main()





