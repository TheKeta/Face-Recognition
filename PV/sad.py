# UPUTSTVO: u komandnoj liniji, posle naziva fajla (sad.py) se unese putanja do slike (npr: bp.jpg iz tog foldera)
# neke od funkcija se preuzete iz projekta: http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
import cv2
import dlib
import numpy
import PIL
from PIL import Image
import math
from skimage import io

from resizeimage import resizeimage
import sys

from os import listdir
from os.path import isfile, join

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im, filename):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    ret = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    razlika = ret[15].item(0) - ret[1].item(0)
    if razlika != 250:
        povecaj = 250.0 / razlika
        slika = Image.open(filename)
        wid, heig = slika.size
        sz = int(wid * povecaj), int(heig * povecaj)
        # slika.thumbnail(sz, Image.ANTIALIAS)
        slika = slika.resize(sz, PIL.Image.ANTIALIAS)
        slika.save(filename)

        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                             im.shape[0] * SCALE_FACTOR))
        ret = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    # pomeranje lica gore levo
    pom_gore = ret[18].item(1)  # leva obrva
    # prolazimo kroz obrve da nadjemo najnizu tacku
    for idx in range(17, 26, 1):
        if ret[idx].item(1) < pom_gore:
            pom_gore = ret[idx].item(1)

    pom_levo = ret[0].item(0)  # levi obraz
    for tacka in ret:
        tacka.itemset(0, tacka.item(0) - pom_levo)
        tacka.itemset(1, tacka.item(1) - pom_gore)

    return ret, im


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s, im = get_landmarks(im, fname)

    return im, s


onlyfiles = [f for f in listdir("examples") if isfile(join("examples", f))]
# mapa svih example slika, key je ime slike a sadrzaj lista [im, landmark]
example_slike = {}
for fajl in onlyfiles:
    el1, el2 = read_im_and_landmarks("examples/" + fajl)
    lista = [el1, el2]
    example_slike[fajl] = el2

im1, landmarks1 = read_im_and_landmarks(sys.argv[1])

matrica_org = []
for idx in range(0, 66, 1):
    for idx2 in range(idx + 1, 67, 1):
        pom = (math.sqrt(abs(landmarks1[idx].item(0) - landmarks1[idx2].item(0)) ** 2 +
                         abs(landmarks1[idx].item(1) - landmarks1[idx2].item(1)) ** 2))
        matrica_org.append(pom)
# primena nearest neigbour alg.
trenutno_najbolji = ""
drugi_najbolji = ""
treci_najbolji = ""
dozvola = 0
razlika = 0
najmanja_razlika = -1
druga_razlika = -1
treca_razlika = -1
for lice in example_slike.keys():
    if dozvola == 0:
        trenutno_najbolji = lice
        dozvola = 1
    matrica_fake = []
    for idx in range(0, 66, 1):
        for idx2 in range(idx + 1, 67, 1):
            po = math.sqrt(abs(example_slike[lice][idx].item(0) - example_slike[lice][idx2].item(0)) ** 2 +
                          abs(example_slike[lice][idx].item(1) - example_slike[lice][idx2].item(1)) ** 2)
            matrica_fake.append(po)
    for index in range(0,2211,1):
        razlika += abs(matrica_fake[index] - matrica_org[index])
    if najmanja_razlika == -1:
        trenutno_najbolji = lice
        najmanja_razlika = razlika
    elif razlika < najmanja_razlika:
        treca_razlika = druga_razlika
        treci_najbolji = drugi_najbolji
        drugi_najbolji = trenutno_najbolji
        druga_razlika = najmanja_razlika
        najmanja_razlika = razlika
        trenutno_najbolji = lice
    elif druga_razlika == -1:
        drugi_najbolji = lice
        druga_razlika = razlika
    elif razlika < druga_razlika:
        treca_razlika = druga_razlika
        treci_najbolji = drugi_najbolji
        drugi_najbolji = lice
        druga_razlika = razlika
    elif treca_razlika == -1:
        treci_najbolji = lice
        treca_razlika = razlika
    elif razlika < treca_razlika:
        treci_najbolji = lice
        treca_razlika = razlika
    print(lice, razlika)
    razlika = 0
print ("Prvi:" + trenutno_najbolji)
print ("Drugi:" + drugi_najbolji)
print ("Treci:" + treci_najbolji)
trenutno_najbolji.replace("1", "")
drugi_najbolji.replace("1", "")
treci_najbolji.replace("1", "")
najbolji = trenutno_najbolji
if (drugi_najbolji is treci_najbolji):
    najbolji = drugi_najbolji

win = dlib.image_window()
slika = Image.open("examples/" + najbolji)
sl = io.imread("examples/" + najbolji)
# slika.show()
win.clear_overlay()
win.set_image(sl)
dlib.hit_enter_to_continue()
# print (landmarks1)
