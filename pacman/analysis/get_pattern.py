import urllib.request
import re

req = urllib.request.Request('https://strategywiki.org/wiki/Pac-Man/Walkthrough', headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    idx = html.find('Goodrich and Butcher Cherry pattern')
    if idx != -1:
        print(html[idx:idx+1500])
    else:
        print('Not found, finding images:')
        for img in re.findall(r'<img[^>]+src="([^">]+)"', html):
            if 'pattern' in img.lower() or 'cherry' in img.lower():
                print(img)
except Exception as e:
    print('Error:', e)
