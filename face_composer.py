# face_composer.py
import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

OUT_DIR = "generated_faces"
os.makedirs(OUT_DIR, exist_ok=True)

CANVAS_W = 256
CANVAS_H = 256

def random_color(hue_range=(0, 360), sat_range=(30, 90), light_range=(30, 80)):
    # simple HSL -> RGB approximation using HSV via PIL-compatible tuple
    h = random.uniform(*hue_range)
    s = random.uniform(*sat_range) / 100.0
    v = random.uniform(*light_range) / 100.0
    # use numpy to convert hsv->rgb
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return (int(r*255), int(g*255), int(b*255), 255)

def draw_face_base(size=(CANVAS_W, CANVAS_H), skin_color=None):
    w,h = size
    im = Image.new("RGBA", size, (0,0,0,0))
    draw = ImageDraw.Draw(im)
    if skin_color is None:
        skin_color = random_color(hue_range=(10,40), sat_range=(30,60), light_range=(50,90))
    # face ellipse
    face_bbox = [w*0.12, h*0.08, w*0.88, h*0.9]
    draw.ellipse(face_bbox, fill=skin_color)
    # subtle shading
    shade = Image.new("RGBA", size, (0,0,0,0))
    sd = ImageDraw.Draw(shade)
    sd.ellipse([w*0.2, h*0.2, w*0.85, h*0.95], fill=(0,0,0,20))
    im = Image.alpha_composite(im, shade)
    return im

def draw_eye(size=(60,36), eye_type=None):
    w,h = size
    im = Image.new("RGBA", (w,h), (0,0,0,0))
    draw = ImageDraw.Draw(im)
    # white of eye
    draw.ellipse([0,h*0.1,w,h*0.9], fill=(255,255,255,255))
    # iris
    iris_r = int(min(w,h) * 0.25)
    iris_x = w*0.5 + random.randint(-6,6)
    iris_y = h*0.5 + random.randint(-3,3)
    iris_color = random_color(hue_range=(0,360), sat_range=(30,100), light_range=(20,70))
    draw.ellipse([iris_x-iris_r, iris_y-iris_r, iris_x+iris_r, iris_y+iris_r], fill=iris_color)
    # pupil
    pupil_r = int(iris_r*0.45)
    draw.ellipse([iris_x-pupil_r, iris_y-pupil_r, iris_x+pupil_r, iris_y+pupil_r], fill=(0,0,0,255))
    # eyebrow (simple line/arc)
    brow_color = (20,20,20,255)
    brow_h = int(h*0.25)
    draw.line([(w*0.15, brow_h), (w*0.85, brow_h - random.randint(0,6))], fill=brow_color, width=3)
    # small highlight
    draw.ellipse([iris_x-iris_r*0.5, iris_y-iris_r*0.8, iris_x-iris_r*0.3, iris_y-iris_r*0.6], fill=(255,255,255,200))
    return im

def draw_nose(size=(40,50)):
    w,h = size
    im = Image.new("RGBA", (w,h), (0,0,0,0))
    draw = ImageDraw.Draw(im)
    # triangle/soft nose
    x0,y0 = w*0.5, h*0.15
    x1,y1 = w*0.15, h*0.9
    x2,y2 = w*0.85, h*0.9
    draw.polygon([(x0,y0),(x1,y1),(x2,y2)], fill=(200,160,140,120))
    # soft blur to blend
    im = im.filter(ImageFilter.GaussianBlur(radius=2))
    return im

def draw_mouth(size=(120,40), expression=None):
    w,h = size
    im = Image.new("RGBA", (w,h), (0,0,0,0))
    draw = ImageDraw.Draw(im)
    # expression: smile, frown, neutral
    if expression is None:
        expression = random.choice(["smile","neutral","frown","open"])
    if expression == "smile":
        draw.arc([10, -10, w-10, h*2], start=200, end=340, fill=(180,30,60,255), width=5)
    elif expression == "frown":
        draw.arc([10, -10, w-10, h*2], start=20, end=160, fill=(180,30,60,255), width=5)
    elif expression == "neutral":
        draw.line([(10,h*0.5),(w-10,h*0.5)], fill=(170,50,70,255), width=6)
    else: # open mouth
        draw.ellipse([10,5,w-10,h-5], fill=(170,40,60,255))
        draw.ellipse([w*0.25,h*0.25,w*0.75,h*0.9], fill=(0,0,0,255))
    return im

def draw_hair(size=(CANVAS_W, CANVAS_H), style=None):
    w,h = size
    im = Image.new("RGBA", (w,h), (0,0,0,0))
    draw = ImageDraw.Draw(im)
    if style is None:
        style = random.choice(["short","long","bald","curly","fringe"])
    hair_color = random_color(hue_range=(0,50), sat_range=(30,80), light_range=(5,40))
    if style == "bald":
        return im
    elif style == "short":
        draw.ellipse([w*0.1, -h*0.15, w*0.9, h*0.45], fill=hair_color)
    elif style == "long":
        draw.rectangle([w*0.05, h*0.05, w*0.95, h*0.9], fill=hair_color)
        im = im.filter(ImageFilter.GaussianBlur(radius=2))
    elif style == "fringe":
        draw.rectangle([w*0.05, -10, w*0.95, h*0.25], fill=hair_color)
    elif style == "curly":
        for i in range(18):
            r = random.randint(20,50)
            cx = random.randint(0,w)
            cy = random.randint(-20,int(h*0.35))
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=hair_color)
    im = im.filter(ImageFilter.GaussianBlur(radius=1.2))
    return im

def paste_with_center(base, part, center_xy):
    px, py = int(center_xy[0] - part.width/2), int(center_xy[1] - part.height/2)
    base.alpha_composite(part, (px, py))

def compose_face(seed=None, save_path=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255,255,255,255))
    face = draw_face_base()
    paste_with_center(canvas, face, (CANVAS_W/2, CANVAS_H/2))
    hair = draw_hair()
    canvas = Image.alpha_composite(canvas, hair)
    eye_w, eye_h = 56, 36
    left_eye = draw_eye((eye_w, eye_h))
    right_eye = left_eye.copy()
    left_eye = left_eye.rotate(random.uniform(-8,8), resample=Image.BICUBIC, expand=True)
    right_eye = right_eye.rotate(random.uniform(-6,6), resample=Image.BICUBIC, expand=True)
    eye_y = CANVAS_H * 0.42 + random.randint(-6,6)
    eye_x_left = CANVAS_W * 0.34 + random.randint(-8,8)
    eye_x_right = CANVAS_W * 0.66 + random.randint(-8,8)
    paste_with_center(canvas, left_eye, (eye_x_left, eye_y))
    paste_with_center(canvas, right_eye, (eye_x_right, eye_y))
    nose = draw_nose((46,60)).rotate(random.uniform(-4,4), expand=True)
    paste_with_center(canvas, nose, (CANVAS_W*0.5 + random.randint(-4,4), CANVAS_H*0.56 + random.randint(-6,6)))
    mouth = draw_mouth((110,44))
    mouth = mouth.rotate(random.uniform(-6,6), expand=True)
    paste_with_center(canvas, mouth, (CANVAS_W*0.5, CANVAS_H*0.72 + random.randint(-6,6)))
    blush = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0,0,0,0))
    bd = ImageDraw.Draw(blush)
    bx = CANVAS_W*0.25; by = CANVAS_H*0.62
    bd.ellipse([bx-12,by-8,bx+12,by+8], fill=(255,100,120,80))
    bd.ellipse([CANVAS_W-bx-12,by-8,CANVAS_W-bx+12,by+8], fill=(255,100,120,80))
    canvas = Image.alpha_composite(canvas, blush)
    canvas = canvas.filter(ImageFilter.SMOOTH)
    if save_path:
        canvas.convert("RGB").save(save_path, quality=95)
    return canvas

def batch_generate(n=10, out_dir=OUT_DIR):
    for i in range(n):
        seed = random.randint(0, 2**30)
        out = os.path.join(out_dir, f"face_{i:03d}_s{seed}.png")
        im = compose_face(seed=seed, save_path=out)
        print("Saved", out)

if __name__ == "__main__":
    batch_generate(12)
    print("Done. Check the generated_faces/ folder.")
