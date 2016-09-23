from PIL import Image
import sys

lena=Image.open(sys.argv[1])

lena.rotate(180).save("ans2.png")