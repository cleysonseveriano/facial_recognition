import cv2
import face_recognition
import mediapipe as mp
import requests
import glob
# print(
#     glob.glob("C:/Users/cleys/OneDrive/Área de Trabalho/facial_recog/images/*.jpg")
# )
# images_jpg = [cv2.imread(file) for file in glob.glob("C:\Users\cleys\OneDrive\Área de Trabalho\facial_recog\images\*.jpg")] 
# images_jpeg = [cv2.imread(file) for file in glob.glob("C:\Users\cleys\OneDrive\Área de Trabalho\facial_recog\images\*.jpeg")]
# images = images_jpg + images_jpeg

# for i in range(len(images)):
#     cv2.imshow("IMG"+str(i), images[i])

encondeds = []
#CADASTRANDO E CODIFICANDO CLEYSON IMAGE
imgCleyson = cv2.imread('images/cleyson.jpeg')
rgbCleyson = cv2.cvtColor(imgCleyson, cv2.COLOR_BGR2RGB)
encodeCleyson = face_recognition.face_encodings(rgbCleyson)[0]
#CADASTRANDO E CODIFICANDO MESSI IMAGE
imgMessi = cv2.imread('images/messi.webp')
rgbMessi = cv2.cvtColor(imgMessi, cv2.COLOR_BGR2RGB)
encodeMessi = face_recognition.face_encodings(rgbMessi)[0]

webcam = cv2.VideoCapture(0)

reconhecimento_rosto = mp.solutions.face_detection 
desenho = mp.solutions.drawing_utils
reconhecedor_rosto = reconhecimento_rosto.FaceDetection()
encodeImg = []
while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break
    imagem = frame
    lista_rostos = reconhecedor_rosto.process(imagem)
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(imagem, rosto)
            rgbImg = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            encodeImg = face_recognition.face_encodings(rgbImg)
            # print(encodeImg)
    cv2.imshow("Rostos na sua webcam", imagem)
    if cv2.waitKey(5) == 27:
        break
webcam.release()
compare = face_recognition.compare_faces(encodeCleyson, encodeImg)

shouldOpen = compare[0]

if shouldOpen:
    print('Enviando request para liberação da catraca do Cleyson')
    requests.post('http://localhost:3000/add', data={'name': 'Cleyson'})

print(shouldOpen)
cv2.destroyAllWindows() 
#COMPARANDO
""" compare = face_recognition.compare_faces([encodeCleyson], encodeMessi)
print('Result: ', compare) """


""" cv2.imshow('Cleyson image', imgCleyson)
cv2.imshow('Messi image', imgMessi)
cv2.waitKey(0) """