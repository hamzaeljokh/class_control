import numpy as np
import cv2

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.optimizers import Adam

from os import listdir
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K


class dnn():
    detection_threshold = 0.95
    reco_threshold = 0.5
    Frame_number = 15
    detection_DNN = "CAFFE"
    etudiant_pictures = "imgs/"
    affichage = True

    def __init__(self, dir_formation):
        K.clear_session()
        self.etudiants = list()
        self.labels = list()
        self.classes = set()


        # onehot encoder
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        self.dir_formation = dir_formation

        self.prepros_data()

    def train_classifier(self, ep=10000):
        c = self.classifier()
        c.fit(x=self.etudiants, y=self.onehot_encoded_labels, epochs=ep)
        c.save('models/classifier_'+self.dir_formation+'.h5')
        K.clear_session()

    def prepros_data(self):
        #K.clear_session()
        model = self.loadVggFaceModel()
        modelFile = "C:/Users/HELL/Desktop/models/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "C:/Users/HELL/Desktop/models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        nb_imgs = 0
        for etu_dir in listdir(self.etudiant_pictures + self.dir_formation + "/"):
            self.classes.add(etu_dir)
            for file in listdir(self.etudiant_pictures + self.dir_formation + "/" + etu_dir + "/"):
                imgpath = self.etudiant_pictures + self.dir_formation + "/" + etu_dir + "/" + file
                nb_imgs = nb_imgs +1
                img = cv2.imread(imgpath)
                img = img_to_array(img)
                img = img.astype(np.uint8)
                faces = self.detectFaceOpenCVDnn(net, img)

                if len(faces) == 1: # une image doit contenir un et un seul visage.
                    (x1, y1, x2, y2) = faces[0]
                    detected_face = img[int(y1):int(y2), int(x1):int(x2)]  # crop detected face
                    detected_face = cv2.resize(detected_face, (224, 224))  # resize to 224x224
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255

                    e_presentaion = model.predict(img_pixels)[0, :]
                    self.etudiants.append(e_presentaion)
                    self.labels.append(etu_dir)

        print("######## A partir de", nb_imgs, "images, il y a", len(self.etudiants), "images valides (seul visage)")

        integer_encoded = self.label_encoder.fit_transform(self.labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.onehot_encoded_labels = self.onehot_encoder.fit_transform(integer_encoded)

        self.etudiants = np.array(self.etudiants).reshape(-1, 2622)
        #del model
        K.clear_session()

    def start_controle(self):
        #K.clear_session()
        etudiant_detecter = set()

        model = self.loadVggFaceModel()

        # vgg-model- keras initialisation
        if self.detection_DNN == "CAFFE":
            modelFile = "C:/Users/HELL/Desktop/models/res10_300x300_ssd_iter_140000.caffemodel"
            configFile = "C:/Users/HELL/Desktop/models/deploy.prototxt"
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = "C:/Users/HELL/Desktop/models/opencv_face_detector_uint8.pb"
            configFile = "C:/Users/HELL/Desktop/models/opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile)

        classifier = load_model('models/classifier_' + self.dir_formation + '.h5')
        #classifier._make_predict_function()

        # Start camera
        cap = cv2.VideoCapture(0)  # webcam
        for i in range(self.Frame_number):
            try:
                ret, img = cap.read()
                faces = self.detectFaceOpenCVDnn(net, img)

                for (x1, y1, x2, y2) in faces:
                    try:
                        #print((x1, y1, x2, y2))
                        detected_face = img[int(y1):int(y2), int(x1):int(x2)]  # crop detected face
                        detected_face = cv2.resize(detected_face, (224, 224))  # resize to 224x224
                        img_pixels = image.img_to_array(detected_face)
                        img_pixels = np.expand_dims(img_pixels, axis=0)
                        img_pixels /= 255

                        captured_representation = model.predict(img_pixels)[0, :]

                        #reconaitre l'etudiant
                        a = classifier.predict(np.array(captured_representation).reshape(-1, 2622))

                        #print("original face detected::",a)
                        a=a.tolist()[0]
                        argmax= max(a)

                        if self.reco_threshold <= argmax:
                            a=[[1 if e == argmax else 0 for e in a]]
                            #print("one hot face detected::", a)
                            int_incode = self.onehot_encoder.inverse_transform(a)
                            int_incode = int_incode.reshape(len(int_incode), 1)[0]
                            prenom_nom = self.label_encoder.inverse_transform(int_incode)
                            #print("l'étudiant:", prenom_nom[0], "est detecter présent pendant le controle")
                            etudiant_detecter.add(prenom_nom[0])
                            if (self.affichage):
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # draw rectangle to main image
                                cv2.putText(img, prenom_nom[0]+':'+str(argmax), (int(x1 + 7), int(y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (255, 0, 0), 2)

                    except Exception as e:
                        print("erreur : reconnaissance")
                        print(e.args)
                        pass
            except:
                print("erreur : detection")
                print(e.args)
                pass

            if (self.affichage):
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break


        cap.release()
        cv2.destroyAllWindows()
        K.clear_session()
        return etudiant_detecter

    # Fonction utilitaire
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def detectFaceOpenCVDnn(self, net, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.detection_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])

        #clear_session()
        return bboxes

    def classifier(self):
        m = Sequential()
        m.add(Dense(2048, input_dim=2622, activation='relu'))
        #model.add(Dropout(0.3))

        m.add(Dense(1024, activation='relu'))
        #model.add(Dropout(0.3))

        m.add(Dense(512, activation='relu'))
        #model.add(Dropout(0.3))

        m.add(Dense(128, activation='relu'))

        m.add(Dense(32, activation='relu'))

        m.add(Dense(16, activation='relu'))

        m.add(Dense(len(self.classes), kernel_initializer='normal', activation='sigmoid'))

        # Compile model
        ad = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        m.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        return m

    def loadVggFaceModel(self):
        #K.clear_session()
        modelv = Sequential()
        modelv.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        modelv.add(Convolution2D(64, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(64, (3, 3), activation='relu'))
        modelv.add(MaxPooling2D((2, 2), strides=(2, 2)))

        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(128, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(128, (3, 3), activation='relu'))
        modelv.add(MaxPooling2D((2, 2), strides=(2, 2)))

        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(256, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(256, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(256, (3, 3), activation='relu'))
        modelv.add(MaxPooling2D((2, 2), strides=(2, 2)))

        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(512, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(512, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(512, (3, 3), activation='relu'))
        modelv.add(MaxPooling2D((2, 2), strides=(2, 2)))

        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(512, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(512, (3, 3), activation='relu'))
        modelv.add(ZeroPadding2D((1, 1)))
        modelv.add(Convolution2D(512, (3, 3), activation='relu'))
        modelv.add(MaxPooling2D((2, 2), strides=(2, 2)))

        modelv.add(Convolution2D(4096, (7, 7), activation='relu'))
        modelv.add(Dropout(0.5))
        modelv.add(Convolution2D(4096, (1, 1), activation='relu'))
        modelv.add(Dropout(0.5))
        modelv.add(Convolution2D(2622, (1, 1)))
        modelv.add(Flatten())
        modelv.add(Activation('softmax'))

        # https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        modelv.load_weights("C:/Users/HELL/Desktop/models/vgg_face_weights.h5")
        modelv._make_predict_function()
        vgg_face_descriptor = Model(inputs=modelv.layers[0].input, outputs=modelv.layers[-2].output)
        return vgg_face_descriptor

