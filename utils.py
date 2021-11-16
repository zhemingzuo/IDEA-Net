import os
import cv2
import csv
import numpy as np
import models.dsa_gam
import scipy.io as sio


def Bbox(thresh_bbox, dsa_gam_area):
    """ Box out the foreground """
    img_gray_path = "./output/exp_01/DSA-GAM/I_overline/0.jpg"
    csv_file_name = "./input/train.csv"
    li = []

    with open(csv_file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            li.append([row])

    a = li[0]
    b = a[0]
    w = b[0]
    img_path = os.path.join(w)
    bounding_box_area = dsa_gam_area
    img = cv2.imread(img_gray_path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = 1-img_gray
    cv2.imwrite(img_gray_path, img_gray)
    a = img.shape[0]
    b = img.shape[1]
    img2 = cv2.imread(img_path)
    img3 = cv2.imread(img_path)
    c = img2.shape[0]
    d = img2.shape[1]
    q = c/a
    g = d/b
    ret, thresh = cv2.threshold(img_gray, thresh_bbox, 255, cv2.THRESH_BINARY)
    img, contours, hierarchy = models.dsa_gam.BBOX(thresh)
    rect_x = []
    rect_area = []

    for cnt in contours:
        i = 0
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h

        if bounding_box_area < w*h < 50000:
            x2 = x + w
            y2 = y + h
            rect_x.append((x, y, x2, y2))
            rect_area.append(area)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            x = x * g
            x = int(x)
            y = y * q
            y = int(y)
            w = w * g
            w = int(w)
            h = h * q
            h = int(h)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0,255, 0), 2)

    if len(rect_x) == 0:
        denoised_img_path = './output/exp_01/DSA-EDM/' + '0.png'
        cv2.imwrite(denoised_img_path, img3)
    else:
        a, b, c, d = min(rect_x, key=lambda x:x[0])
        a1, b1, c1, d1 = min(rect_x, key=lambda x:x[1])
        a2, b2, c2, d2 = max(rect_x, key=lambda x:x[2])
        a3, b3, c3, d3 = max(rect_x, key=lambda x:x[3])
        a = int(a * g)
        b1 = int(b1 * q)
        c2 = int(c2 * g)
        d3 = int(d3 * q)
        denoised_img_path = "./output/exp_01/DSA-EDM/" + "_" + str(b1) + "_" + str(d3) + "_" + str(a) + "_" + str(c2) + ".png"
        cv2.imwrite(denoised_img_path, img3[b1:d3, a:c2])

    print("xxxxxxxxxxxxxxxxxxxxxxx")
    print(denoised_img_path)
    print("xxxxxxxxxxxxxxxxxxxxxxx")

    return denoised_img_path


def add_gaussian_noise(img, model_path, sigma_val):

    index = model_path.rfind("/")

    if sigma_val > 0:
        noise = np.random.normal(scale=sigma_val / 255., size=img.shape).astype(np.float32)
        sio.savemat(model_path[0:index] + '/noise.mat', {'noise': noise})
        noisy_img = (img + noise).astype(np.float32)
    else:
        noisy_img = img.astype(np.float32)

    cv2.imwrite(model_path[0:index] + '/noise-image.png',
                np.squeeze(np.int32(np.clip(noisy_img, 0, 1) * 255.)))

    return noisy_img


def load_np_image(path, is_scale=True):

    print("oooooooooooooooooooooooo")
    print(path)
    print("oooooooooooooooooooooooo")
    img = cv2.imread(path, -1)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    img = np.expand_dims(img, axis=0)

    if is_scale:
        img = np.array(img).astype(np.float32) / 255.

    return img
