import glob

import os
#os.system('python.exe inference2.py ./resultc_5c/model_fine_final.h5 classes_5c.txt imagesc/dalmatian_side3.jpg --top_n 1')


for f in glob.glob("./imagesc/*.jpg"):
    #print(f[2:])
    #os.system('python.exe inference2.py ./resultc_5c/model_fine_final.h5 classes_5c.txt '+ f[2:] +' --top_n 1')
    os.system('python.exe inference2.py ./resultc/model_fine_final.h5 classes.txt '+ f[2:] +' --top_n 1')
    #os.system('python.exe inference2.py ./resultc_6c/model_fine_final.h5 classes_6c.txt '+ f[2:] +' --top_n 1')