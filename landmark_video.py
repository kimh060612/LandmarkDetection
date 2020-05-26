import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import cv2


pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

check = [0 for _ in range(68)]

if __name__ == "__main__":

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)

    VideoPath = "./test/NigeHazi.avi"
    OutVideo = "./test/NigeHaziResult.avi"
    Vid = cv2.VideoCapture(VideoPath)

    frame_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = Vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(frame_width, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(OutVideo, fourcc, 25.0, (1280,720))

    while Vid.isOpened():
        try:
            ret, frame = Vid.read()
            Pri_index = 0
            #input_img = io.imread(frame)
            K = fa.get_landmarks(frame)
            if not K == None :
                preds = K[-1]
                if not len(preds) == 0:
                    for pred_type in pred_types.values():
                        Det = len(check[pred_type.slice])
                        for i in range(Det):
                            frame = cv2.line(frame, (preds[Pri_index + i, 0],preds[Pri_index + i, 1]), (preds[Pri_index + i, 0],preds[Pri_index + i, 1]) , (0, 0, 255), 5) 
                        Pri_index += (Det - 1)
                else :
                    print("Failed")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            out.write(frame)
            cv2.imshow('video', frame)
        except :
            print("Failed")



Vid.release()
out.release()
cv2.destroyAllWindows()