import re

path = r'C:\Users\jayan\.gemini\antigravity\brain\8ce09a10-5109-4657-b8f0-ac452a1f0999\.system_generated\steps\500\content.md'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Look for links containing /bitstreams/ and nearby text or attributes
matches = re.findall(r'<a[^>]*href=[\'"]([^\'"]*bitstreams[^\'"]*)[\'"][^>]*>(.*?)</a>', text, re.DOTALL)
for href, inner in matches:
    clean_inner = re.sub(r'<[^>]+>', '', inner).strip()
    print(f"HREF: {href}\n  TEXT: {clean_inner}\n")

# Also look for aria-label or title
matches_aria = re.findall(r'<a[^>]*href=[\'"]([^\'"]*bitstreams[^\'"]*)[\'"][^>]*title=[\'"]([^\'"]*)[\'"]', text, re.DOTALL)
for href, title in matches_aria:
    print(f"HREF: {href}\n  TITLE: {title}\n")
