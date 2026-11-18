import urllib.request
import re

url = 'https://html.duckduckgo.com/html/?q=Pac-Man+level+1+cherry+pattern+step+by+step'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', html)
    for i, s in enumerate(snippets[:10]):
        clean = re.sub('<[^<]+?>', '', s)
        print(f"[{i}] {clean}")
except Exception as e:
    print('Error:', e)
