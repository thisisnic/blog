#!/usr/bin/env python3
"""Filter blog.xml to create blog-r.xml with only R-tagged posts"""

import xml.etree.ElementTree as ET
import sys
from email.utils import parsedate_to_datetime, format_datetime

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

    # Quarto sets lastBuildDate from the newest post in the full feed, so an
    # update to a non-R post would bump it here too. Recompute it from the
    # newest remaining R post so the R feed only changes when R posts do.
    remaining = channel.findall('item')
    pub_dates = []
    for item in remaining:
        pub_date = item.find('pubDate')
        if pub_date is not None and pub_date.text:
            pub_dates.append(parsedate_to_datetime(pub_date.text))
    last_build = channel.find('lastBuildDate')
    if pub_dates and last_build is not None:
        last_build.text = format_datetime(max(pub_dates), usegmt=True)

    # Point the feed's self link at itself rather than blog.xml
    self_link = channel.find('atom:link', namespaces)
    if self_link is not None and self_link.get('rel') == 'self':
        self_link.set('href', self_link.get('href', '').replace('blog.xml', 'blog-r.xml'))

    # Write filtered RSS
    tree.write('_site/blog-r.xml', encoding='UTF-8', xml_declaration=True)
    print(f"Created blog-r.xml with {len(items) - len(items_to_remove)} R-tagged posts")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
