import re

path = r'C:\Users\jayan\.gemini\antigravity\brain\8ce09a10-5109-4657-b8f0-ac452a1f0999\.system_generated\steps\500\content.md'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

zip_matches = re.findall(r'href=[\'"]([^\'"]*?[A-Za-z0-9_\-\.]+\.zip[^\'"]*?)[\'"]', text, re.IGNORECASE)
print('Found zip links:', len(zip_matches))
for z in set(zip_matches):
    print('  ZIP:', z)

all_links = re.findall(r'href=[\'"]([^\'"]*?bitstream[^\'"]*?)[\'"]', text, re.IGNORECASE)
print('Found bitstream links:', len(all_links))
for b in set(all_links):
    print('  BITSTREAM:', b)

# Search for any filenames mentioning LA
la_files = re.findall(r'([A-Za-z0-9_\-\.]*LA[A-Za-z0-9_\-\.]*\.(?:zip|txt|tar|gz))', text, re.IGNORECASE)
print('Found LA files:', set(la_files))
