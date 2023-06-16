import cv2 as cv


def improve_image():
    image = cv.imread('images/variant-10.jpg')
    # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, result_image = cv.threshold(image, 150, 255, 0)
    cv.imshow('result', result_image)
    cv.imshow('default', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_ref_point():
    # Берем изображение с камеры
    cap = cv.VideoCapture(0)
    # Создаем шаблон, по которому будет находиться соответствие
    templ_img = cv.imread('ref-point.png', 0)
    ret, templ_thresh = cv.threshold(templ_img, 127, 255, 0)
    contours, hierarchy = cv.findContours(templ_thresh, 2, 1)
    templ_cnt = [contours[0], contours[1]]
    flag = False
    fly = cv.imread('fly64.jpg')
    while True:
        x, y, h, w = [], [], [], []
        # Считывем очередной кадр, выполняем преобразования (перевод в ч\б, размытие, пороговая фильтрация)
        ret, video_img = cap.read()
        if flag == True:
            video_img = cv.rotate(video_img, cv.ROTATE_180)
        video_gray = cv.cvtColor(video_img, cv.COLOR_BGR2GRAY)

        img_height, img_width = len(video_gray), len(video_gray[0])
        video_gray = cv.GaussianBlur(video_gray, (7, 7), 0)
        ret, video_thresh = cv.threshold(video_gray, 130, 255, 0)
        contours, hierarchy = cv.findContours(video_thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # Устанавливаем порог соответствия, каждому контуру шаблона ищем соответствующий контур в текущем кадре
        similarity_level = 0.02
        for video_cnt in contours:
            for i in range(len(templ_cnt)):
                ret = cv.matchShapes(templ_cnt[i], video_cnt, 1, 0.0)
                if (ret < similarity_level) & ((len(video_cnt) > 20)):
                    x.append(cv.boundingRect(video_cnt)[0])  # Сохраняем координаты
                    y.append(cv.boundingRect(video_cnt)[1])
                    h.append(cv.boundingRect(video_cnt)[2])
                    w.append(cv.boundingRect(video_cnt)[3])
                    break
        # Если метка найдена, находим ее границы и выводим квадрат
        height = 0
        width = 0
        if len(x) == 2:
            coord_x = min(x)
            coord_y = min(y)
            for i in range(len(h)):
                height += h[i]
                width += w[i]

            # Добавляем муху
            bigger_fly = cv.resize(fly, (width, height))
            for i in range(height):
                for j in range(width):
                    if (bigger_fly[i, j, 0] < 250) & (bigger_fly[i, j, 1] < 250) & (bigger_fly[i, j, 2] < 250):
                        video_img[coord_y + i, coord_x + j] = bigger_fly[i, j]
            # Проверяем на попадание в центр
            coord_y = coord_y + (height // 2)
            coord_x = coord_x + (width // 2)
            if (coord_x > (img_width//2 - 75)) & (coord_x < (img_width//2 + 75)) & (coord_y > (img_height//2 - 75)) & (coord_y < (img_height//2 + 75)):
                flag = not flag

        # Выводим результат
        cv.imshow('canny', video_thresh)
        cv.imshow('default', video_img)

        ch = cv.waitKey(5)
        if ch == 27:
            break
    cap.release()
    cv.destroyAllWindows()


print("Приветствую!")
print("Показать результат первого задания? (y/n)")
if input() == 'y':
    improve_image()
print("Показать результат второго и дополнительного задания? (y/n)")
if input() == 'y':
    find_ref_point()
print("До свидания!")
