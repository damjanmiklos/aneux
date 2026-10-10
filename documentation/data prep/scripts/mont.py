import sys
from PIL import Image
out=sys.argv[1]; fs=sys.argv[2:]
ims=[Image.open(f) for f in fs]; w,h=700,int(700*ims[0].height/ims[0].width)
ims=[im.resize((w,h)) for im in ims]
W=Image.new('RGB',(w*len(ims),h),'white')
for j,im in enumerate(ims): W.paste(im,(w*j,0))
W.save(out)
