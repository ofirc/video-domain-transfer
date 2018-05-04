import sys, cv2

CASCADE_XML='haarcascade_frontalface_default.xml'

def cascade_detect(cascade, image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return cascade.detectMultiScale(
    gray_image,
    scaleFactor = 1.35, # 1.15
    minNeighbors = 5,
    minSize = (30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE    # not suitable for Python 3.5 with OpenCV 3.0
    flags = cv2.CASCADE_SCALE_IMAGE
  )

WIDTH = 178
HEIGHT = 218
ASPECT_RATIO = float(WIDTH) / HEIGHT # resolution of StarGAN CelebA pictures.

def get_new_detection(x, y, w, h):
  w_new = int(w * 1.2)
  h_new = w_new / ASPECT_RATIO
  
  mid = y + h/2
  y_new = int(mid - h_new / 2)
  x_new = int(x-0.1*w)
  h_new = int(h_new)

  return x_new, y_new, w_new, h_new

def detections_draw(image, detections):
  # TODO: protect against false-positive faces (e.g. background objects)
  x_ret = y_ret = w_max = h_max = 0
  for (x, y, w, h) in detections:
    x_t, y_t, w_t, h_t = get_new_detection(x, y, w, h)
    if (w_t*h_t > w_max*h_max):
        x_ret, y_ret = x_t, y_t
        w_max, h_max = w_t, h_t

  return x_ret, y_ret, w_max, h_max

def _main(image_path, result_path):
  cascade_path = CASCADE_XML
  cascade = cv2.CascadeClassifier(cascade_path)
  image = cv2.imread(image_path)
  if image is None:
    print("ERROR: Image did not load.")
    return None

  detections = cascade_detect(cascade, image)
  x, y, w, h = detections_draw(image, detections)

  #print("Found {0} objects!".format(len(detections)))
  if result_path is None:
    cv2.imshow("Objects found", image)
    cv2.waitKey(0)
  else:
    #cv2.imwrite(result_path, image)
    cropped_img = image[y:y+h, x:x+w]
    resized_img = cv2.resize(cropped_img, (WIDTH, HEIGHT))
    cv2.imwrite(result_path, resized_img)

  return w

def main(argv = None):
  if argv is None:
    argv = sys.argv[1:]

  if len(argv) == 0:
    print("Usage: %s <frame> [<output frame>]" % (__file__))
    return -1

  image_path = argv[0]
  result_path = argv[1] if len(argv) > 1 else None

  ret = _main(image_path, result_path)
  if ret is None:
    return -1
  
  return 0


if __name__ == "__main__":
  sys.exit(main())
