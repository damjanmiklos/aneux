import re
s = open('C:/dev/UQ/aneux/documentation/data prep/scripts/crossref.bib', encoding='utf-8').read()
s = s.replace('&amp;', '&')
s = s.replace('&', '\\&')
out = []
for e in re.findall(r'@\w+\{.*?\}\s*(?=@|\Z)', s, flags=re.S):
    e = e.strip()
    head, body = e.split(',', 1)
    body = body.rstrip()
    if body.endswith('}'):
        body = body[:-1]
    fields = re.split(r',\s(?=[A-Za-z]+=)', body.strip())
    out.append(head + ',\n' + ',\n'.join('  ' + f.strip() for f in fields) + '\n}\n')
vtk = '''@book{schroeder2006,
  title={The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics},
  edition={4},
  publisher={Kitware},
  author={Schroeder, Will and Martin, Ken and Lorensen, Bill},
  year={2006},
  isbn={978-1-930934-19-1}
}
'''
open('C:/dev/UQ/aneux/documentation/data prep/dataprep.bib', 'w', encoding='utf-8').write('\n'.join(out) + '\n' + vtk)
print(len(out) + 1)
