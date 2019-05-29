import sys, os
import cv2

imgdir = sys.argv[1]
outname = sys.argv[2]
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video=cv2.VideoWriter(outname,fourcc,15,(320,240))

namelist = os.listdir(imgdir)
namelist = sorted(namelist)
#for idx in range(2, 256, 2):
for imname in namelist:
    if imname > "test_two_faces_0166.jpg":
        break
    #imname = "jiaxin_test_jiaxin_face_%d.jpg"%idx
    impath = os.path.join(imgdir, imname)
    img = cv2.imread(impath)
    print img.shape
    video.write(img)

video.release()
cv2.destroyAllWindows()
