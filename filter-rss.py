#!/usr/bin/env python3
"""Filter blog.xml to create blog-r.xml with only R-tagged posts"""

import xml.etree.ElementTree as ET
import sys

# Register namespaces to preserve them in output
namespaces = {
    'atom': 'http://www.w3.org/2005/Atom',
    'media': 'http://search.yahoo.com/mrss/',
    'content': 'http://purl.org/rss/1.0/modules/content/',
    'dc': 'http://purl.org/dc/elements/1.1/'
}

for prefix, uri in namespaces.items():
    ET.register_namespace(prefix, uri)

try:
    tree = ET.parse('_site/blog.xml')
    root = tree.getroot()
    channel = root.find('channel')

    # Find all items
    items = root.findall('.//item')
    items_to_remove = []

    for item in items:
        # Check if item has an "R" category
        categories = item.findall('category')
        has_r_category = any(cat.text == 'R' for cat in categories)

        if not has_r_category:
            items_to_remove.append(item)

    # Remove items without R category
    for item in items_to_remove:
        channel.remove(item)

    # Write filtered RSS
    tree.write('_site/blog-r.xml', encoding='UTF-8', xml_declaration=True)
    print(f"Created blog-r.xml with {len(items) - len(items_to_remove)} R-tagged posts")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
