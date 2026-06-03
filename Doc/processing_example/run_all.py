import run
import Resize
import Double_thresholding
import Mask_img_SAM
import segment_anything_run
import sam_fish_school

print("PREPROCESSING")
run()
print("RESIZE")
Resize()
print("DOUBLE THRESHOLDING")
Double_thresholding()
print("MASKING BEFORE SAM")
Mask_img_SAM()
print("SAM")
segment_anything_run()
print("EXTRACTING SAM SCHOOLS")
sam_fish_school()

