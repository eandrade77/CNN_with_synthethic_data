from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions
)
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import argparse
import csv
import glob

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('classes')
parser.add_argument('image')
parser.add_argument('--top_n', type=int, default=2)

def main(args):

    # create model
    model = load_model(args.model)

    # load class names
    classes = []
    with open(args.classes, 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))

   
    for f in glob.glob("./imagesc/*.jpg"):

        # load an input image
        #img = image.load_img(args.image, target_size=(299, 299))
        img = image.load_img(f, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # predict
        pred = model.predict(x)[0]
        result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
        result.sort(reverse=True, key=lambda x: x[1])
        for i in range(args.top_n):
            (class_name, prob) = result[i]
            print("Top %d ====================" % (i + 1))
            print("Class name: %s" % (class_name))
            print("Probability: %.2f%%" % (prob))

        print(class_name, args.image)


        image_ = f
        class_ = class_name
        prob_ = prob
        model_ = args.model

        with open('experimento33.csv', mode='a') as file_:
            file_.write("{},{},{},{}".format(image_, class_, prob_, model_))
            file_.write("\n")



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
