
import cv2
import numpy as np
from math import copysign
from os import makedirs, listdir


TEXT_THRESHOLD = 50
BLUR_KERNEL_SIZE_X = 4
BLUR_KERNEL_SIZE_Y = 5
WIDTH_CORRECTION = 4
MAX_WORD_SIZE = 200
HEIGHT_MAX_DIFFER_FROM_AVERAGE = 100
GRID_CORRECTION = 4
HEIGHT_MANUAL_ADJUST = 0


def print_boxes(b_l, col=100):
    img_boxes = filtered.copy()
    for l_b in b_l:
        cv2.rectangle(img_boxes, (l_b[0], l_b[1]), (l_b[0] + l_b[2], l_b[1] + l_b[3]), col, 1)
    show_window(str(b_l), img_boxes)


def is_overlapping(obj, arr):
    c = 0
    for check in arr:
        if obj == check:
            continue
        if obj[0] >= check[0] + check[2] or obj[0] + obj[2] <= check[0]:
            continue
        if obj[1] >= check[1] + check[3] or obj[1] + obj[3] <= check[1]:
            continue
        c += 1
    return c


def is_in(obj, c_list):
    b_x, b_y, b_w, b_h = obj[0], obj[1], obj[2], obj[3]
    for check_box in c_list:
        _x, _y, _w, _h = check_box[0], check_box[1], check_box[2], check_box[3]
        if _x == b_x and _y == b_y and _w == b_w and _h == b_h:
            continue
        if _x <= b_x and _x + _w >= b_x + b_w and _y <= b_y and _y + _h >= b_y + b_h:
            return True
    return False


def show_window(name, win):
    cv2.imshow(name, win)
    cv2.waitKey()
    #cv2.destroyAllWindows()


def calculate_boxes_from_image(image):
    contours, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    boundary_list = []
    for c in contours:
        bb = cv2.boundingRect(c)
        # save all boxes except the one that has the exact dimensions of the image (x, y, width, height)
        if bb[0] == 0 and bb[1] == 0 and bb[2] == filtered.shape[1] and bb[3] == filtered.shape[0]:
            continue
        boundary_list.append(bb)
    boundary_list.sort(key=lambda x: x[1])

    return boundary_list


def remove_isolated(image):
    image_inv = cv2.bitwise_not(image)
    # clear isolated
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)

    hitormiss1 = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1)
    hitormiss2 = cv2.morphologyEx(image_inv, cv2.MORPH_ERODE, kernel2)
    hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
    hitormiss_comp = cv2.bitwise_not(hitormiss)
    return cv2.bitwise_and(image, image, mask=hitormiss_comp)


def mask_mid_value(image):
    av = np.average(image)
    lower = np.array([av - TEXT_THRESHOLD, av - TEXT_THRESHOLD, av - TEXT_THRESHOLD])
    upper = np.array([av + TEXT_THRESHOLD, av + TEXT_THRESHOLD, av + TEXT_THRESHOLD])
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_not(mask)


def blur(image, size_x, size_y):
    return cv2.blur(image, (size_x, size_y))


def calculate_shift(obj, e_list):
    min_delta = 999999
    sign = 1
    for check in e_list:
        min_delta = min(min_delta, abs(check[1] - obj[1]), abs(check[1] + check[3] - obj[1] - obj[3]))
        if min_delta == abs(check[1] - obj[1]):
            sign = copysign(sign, check[1] - obj[1])
        elif min_delta == abs(check[1] + check[3] - obj[1] - obj[3]):
            sign = copysign(sign, check[1] + check[3] - obj[1] - obj[3])
    return min_delta * sign


def recalculate(image, bboxes, _grid):
    _height, _width = image.shape
    mask1 = np.zeros((_height, _width), np.uint8)
    out = []
    for box in bboxes:
        cv2.rectangle(mask1, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), 255, -1)
    for box in grid:
        mask2 = np.zeros((_height, _width), np.uint8)
        cv2.rectangle(mask2, (0, box[1]), (_width, box[1] + box[3]), 255, -1)
        mask = cv2.bitwise_and(mask1, mask2)
        masked_data = cv2.bitwise_and(image, image, mask=mask)
        for thing in calculate_boxes_from_image(masked_data):
            out.append(thing)
    return out


def cleanup_boxes(b_list):
    bb_copy = b_list.copy()
    for box in bb_copy:
        if av_height + HEIGHT_MAX_DIFFER_FROM_AVERAGE < box[3] or (8 > box[3] and 8 > box[2]) or box[2] > MAX_WORD_SIZE:
            b_list.remove(box)
            continue
        if is_in((box[0], box[1] + GRID_CORRECTION // 2, box[2] - 1, box[3] - 1), b_list):
            b_list.remove(box)


img = cv2.imread("/home/cream_cat/Изображения/Y0501_224108.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# debug
# show_window("image", img)

filtered = mask_mid_value(img)
filtered = remove_isolated(filtered)
blurred = blur(filtered, BLUR_KERNEL_SIZE_X, BLUR_KERNEL_SIZE_Y)

bb_list = calculate_boxes_from_image(blurred)
av_height = sum(i[3] for i in bb_list) // len(bb_list)
cleanup_boxes(bb_list)

heights = []
widths = []
heights_all = []
widths_all = []
for box in bb_list:
    heights_all.append(box[3])
    widths_all.append(box[2])
    if 1.5 < box[3] / box[2] < 3:
        heights.append(box[3])
        widths.append(box[2])
av_height = sum(heights_all) // len(heights_all)
av_width = sum(widths_all) // len(widths_all)
if len(heights) > 1:
    height = sorted(heights)[-2] - HEIGHT_MANUAL_ADJUST
    width = sorted(widths)[-2] - WIDTH_CORRECTION
elif len(heights) == 1:
    height = heights[0] - HEIGHT_MANUAL_ADJUST
    width = widths[0] - WIDTH_CORRECTION // 2
else:
    height = round(av_width * 1.8) - HEIGHT_MANUAL_ADJUST
    width = av_width - WIDTH_CORRECTION

bb_list = list(filter(lambda x:  x[2] > width, bb_list))
print_boxes(bb_list)

sorted_list = sorted(bb_list, key=lambda x: x[0])

min_x = sorted_list[0][0]
edge = [box for box in sorted_list if width + min_x - WIDTH_CORRECTION > box[0]]
width = width - 1
max_distance = 0
prev_cord = None
# recalculate height approximation
for box in edge:
    if prev_cord and box[1] - prev_cord < height:
        continue
    if not prev_cord or box[1] - prev_cord > height + height // 2:
        prev_cord = box[1]
        continue
    max_distance = max(max_distance, box[1] - prev_cord)
    prev_cord = box[1]
if not max_distance:
    max_distance = height
height = max_distance

# generate grid
edge = sorted(edge, key=lambda x: x[1])
grid = [(edge[0][0], edge[0][1], width, height)]
for box in edge:
    x, y, w, h = box[0] + GRID_CORRECTION, box[1] + GRID_CORRECTION, box[2] - GRID_CORRECTION, box[3] - GRID_CORRECTION
    if not is_in((x, y, 1, h), grid):
        if is_overlapping(box, grid):
            grid.append((grid[-1][0], grid[-1][1] + height, width, height))
            continue
        grid.append((box[0], box[1], width, height))

sorted_y_list = sorted(bb_list, key=lambda x: x[1])
# fill lines

remapped = []
new = []
for box in sorted_y_list:
    if calculate_shift((box[0], box[1], width, height), grid) > height * 2:
        continue
    if box[3] > (height - GRID_CORRECTION) * 2:
        to_recalculate = []
        for check in sorted_y_list:
            if box == check:
                continue
            if box[0] >= check[0] + check[2] or box[0] + box[2] <= check[0]:
                continue
            if box[1] >= check[1] + check[3] or box[1] + box[3] <= check[1]:
                continue
            to_recalculate.append(check)
        to_recalculate.append(box)
        for i in to_recalculate:
            remapped.append(i)
        for i in recalculate(filtered, to_recalculate, grid):
            new.append(i)
# chonky
cleanup_boxes(new)
for box in remapped:
    sorted_y_list.remove(box)
for box in new:
    sorted_y_list.append(box)
sorted_y_list = sorted(sorted_y_list, key=lambda x: x[1])
shifted_box_list = []
words = []
for box in sorted_y_list:
    x, y, w, h = box[0] + GRID_CORRECTION // 2, box[1] + GRID_CORRECTION // 2, box[2] - GRID_CORRECTION, box[3] - GRID_CORRECTION
    if calculate_shift((box[0], box[1], width, height), grid) > height * 2:
        continue
    if not is_in((x, y, 1, h), shifted_box_list) or round(box[1] + calculate_shift((box[0], box[1], width, height), grid)) < height / 3:
        box = (box[0], round(box[1] + calculate_shift((box[0], box[1], width, height), grid)), box[2], box[3])
        if is_overlapping(box, shifted_box_list):
            parent = False
            for check in shifted_box_list:
                x_c, y_c, w_c, h_c = check[0] + 2, check[1] + 2, check[2] - 5, check[3] - 5
                if is_in((x_c, y_c, w_c, h_c), [box]):
                    shifted_box_list.remove(check)
                    parent = True
            if not parent:
                continue
        inboxes = (box[2] + GRID_CORRECTION) // width
        words.append(inboxes)
        x = box[0]
        for _ in range(inboxes):
            shifted_box_list.append((x, box[1], width, height))
            x += width

print_boxes(shifted_box_list, 100)

# makedirs("./output", exist_ok=True)
# if listdir("./output"):
#     numbers = [int(i[:-4]) for i in listdir("./output")]
#     c = sorted(numbers)[-1] + 1 if listdir("./output") else 0
# else:
#     c = 0
# print(c)
# for box in shifted_box_list:
#     region = filtered[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
#     c += 1
#     cv2.imwrite(f"./output/{c}.png", region)
