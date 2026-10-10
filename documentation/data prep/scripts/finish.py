"""Trim white margins of the chapter figures; protect acronyms in the bib from APA lower-casing."""
import glob, re
from PIL import Image, ImageChops

F = 'C:/dev/UQ/aneux/documentation/data prep/figures'
for f in glob.glob(F + '/*.png'):
    im = Image.open(f).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).convert('L').point(lambda v: 255 if v > 8 else 0).getbbox()
    if box:
        pad = 12
        box = (max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad))
        im.crop(box).save(f)
        print(f.split('/')[-1], im.size, '->', (box[2] - box[0], box[3] - box[1]))

B = 'C:/dev/UQ/aneux/documentation/data prep/dataprep.bib'
s = open(B, encoding='utf-8').read()
words = ['VTK', 'AneuX', 'CFD', 'ECCV', 'SIGGRAPH', 'Python', 'PyVista', 'SciPy', '3D', 'MATCH', 'SplineCNN', 'B-Spline',
         'Vascular Modeling Toolkit', 'Visualization Toolkit', 'Voronoi', 'Eurographics/ACM', 'International Aneurysm CFD Challenge']
def protect(m):
    t = m.group(2)
    for w in sorted(words, key=len, reverse=True):
        t = re.sub(r'(?<![{\w])' + re.escape(w) + r'(?![\w}])', '{' + w + '}', t)
    return m.group(1) + t + m.group(3)
s = re.sub(r'(\n  title ?= ?\{)(.*?)(\},?\n)', protect, s)
s = re.sub(r'(\n  booktitle ?= ?\{)(.*?)(\},?\n)', protect, s)
open(B, 'w', encoding='utf-8').write(s)
print(re.findall(r'\n  title ?= ?\{(.*?)\},?\n', s)[:40])
