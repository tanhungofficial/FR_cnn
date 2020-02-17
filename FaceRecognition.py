import  os
import sys
import cv2
import numpy as np
from keras import  models
from keras.layers import Dense
from keras import  optimizers
from scipy.sparse import coo_matrix
from PyQt5.QtWidgets import  QDialog,QTextEdit,QApplication,QLabel
from PyQt5.QtWidgets import QGroupBox,QVBoxLayout,QHBoxLayout,QPushButton,QGridLayout,QFileDialog
from PyQt5 import QtGui,QtCore
import datetime
from sklearn.decomposition import PCA
class myWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.left=200
        self.top=100
        self.width=500
        self.height=400
        self.title="Face Detection"
        self.img_path= "icon\default.png"
        self.network= NetWork()
        self.initWindow()
        self.show()

    def initWindow(self):
        self.setGeometry(QtCore.QRect(self.left,self.top,self.width,self.height))
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon("window.png"))
        nameLabel=QLabel("Name")
        self.nameText= QTextEdit()
        self.nameText.setFont(QtGui.QFont('sanserif',17))
        self.image= QLabel(self)
        self.image.setGeometry(QtCore.QRect(0,0,100,100))
        self.image.setPixmap(QtGui.QPixmap(self.img_path))
        hbox=QHBoxLayout()
        gbox=QGroupBox("")
        gridLayout = QGridLayout()

        detect_button = QPushButton('Detection', self)
        detect_button.setGeometry(QtCore.QRect(50, 50, 250, 150))
        detect_button.setIcon(QtGui.QIcon('icon\detection.png'))
        detect_button.setMinimumHeight(50)
        detect_button.clicked.connect(self.clickDetection)
        gridLayout.addWidget(detect_button, 0, 0)

        init_button = QPushButton('Initialize', self)
        init_button.setGeometry(QtCore.QRect(50, 50, 250, 150))
        init_button.setIcon(QtGui.QIcon('icon\initalize.png'))
        init_button.setMinimumHeight(50)
        init_button.clicked.connect(self.clickInitialize)
        gridLayout.addWidget(init_button, 0, 1)

        add_button = QPushButton('Camera', self)
        add_button.setGeometry(QtCore.QRect(50, 50, 250, 150))
        add_button.setIcon(QtGui.QIcon('icon\\camera.png'))
        add_button.setMinimumHeight(50)
        add_button.clicked.connect(self.clickAdd)
        gridLayout.addWidget(add_button, 1, 0)

        train_button = QPushButton('Traning', self)
        train_button.setGeometry(QtCore.QRect(50, 50, 250, 150))
        train_button.setIcon(QtGui.QIcon('icon\\training.png'))
        train_button.setMinimumHeight(50)
        train_button.clicked.connect(self.clickTraining)
        gridLayout.addWidget(train_button, 1, 1)

        gbox.setLayout(gridLayout)
        vbox=QVBoxLayout()
        vbox.addWidget(nameLabel)
        vbox.addWidget(self.nameText)
        vbox.addWidget(detect_button)
        vbox.addWidget(init_button)
        vbox.addWidget(add_button)
        vbox.addWidget(train_button)
        gbox1= QGroupBox("Control")
        gbox1.setLayout(vbox).
        hbox.addWidget(self.image)
        hbox.addWidget(gbox1)
        self.setLayout(hbox)
    def clickDetection(self):
        try:
            self.img_path=QFileDialog.getOpenFileName(caption="Select Image")[0]
            result=self.network.nameList[self.network.predict(self.img_path)]
            self.nameText.setText(result)
            self.img_path="image_detection.jpg"
            self.image.setPixmap(QtGui.QPixmap(self.img_path))
        except:
            pass
    def clickInitialize(self):
        self.network.initializeDataBase()
        self.nameText.setText("Initialization completed!")
    def clickAdd(self):
        #self.network.addDataFromCamera(str(self.nameText.toPlainText()))
        try:
            self.network.recognitionFromCamera()
            result=self.network.predict(path='live_image.jpg')
            self.nameText.setText(self.network.nameList[result])
            self.image.setPixmap(QtGui.QPixmap("image_detection.jpg"))
        except:
            pass
    def clickTraining(self):
        self.network.trainingModel()
        self.nameText.setText("Training completed!")
class NetWork():
    def __init__(self):
        self.nameList=[]
        self.label=[]
        self.data=[]
        self.result=0
        self.initializeDataTrain()
    def convertLabel(self,label):
        N = len(label)
        C = np.max(label) + 1
        label_cvt = coo_matrix((np.ones_like(label), (np.arange(N), label)), shape=(N, C)).toarray()
        return label_cvt
    def initializeDataBase(self,path='database', color=1):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        LDR = os.listdir(path)
        for ldr in LDR:
            try:
                img_path = os.path.join(path, ldr)
                img = cv2.imread(img_path, color)
                faces = face_cascade.detectMultiScale(img)
                for (x, y, width, height) in faces:
                    if width*height>4000:
                        img_face = img[y:y + height, x:x + width, :]
                        img_face= cv2.resize(img_face,(256,256))
                        cv2.imwrite(img_path, img_face)
            except:
                pass
    def initializeDataTrain(self,path="database"):
        LDR = os.listdir(path)
        IMG = []
        LABEL = []
        NAME = []
        label = -1
        name = ""
        for ldr in LDR:
            try:
                img_path = os.path.join(path, ldr)
                list = ldr.split('_')
                if name != list[0].upper():
                    name = list[0]
                    NAME.append(name.upper())
                    label+=1
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (64,64))
                img= img.reshape(4096,)/255
                IMG.append(img)
                LABEL.append(label)
            except:
                pass
        self.data=np.array(IMG)
        self.label=np.array(LABEL)
        self.nameList=NAME
    def PCA(self):
        pca=PCA(n_components=45)
        pca.fit(self.data)
        U = pca.components_
    def trainingModel(self,k_pca=60, epoch=150, batch_size=10, lr=0.01, loss="categorical_crossentropy"):
        self.initializeDataTrain()
        labelTrain= self.convertLabel(self.label)
        d = self.data.shape[1]
        N = self.data.shape[0]
        C = labelTrain.shape[1]
        print(self.data.shape)
        print(labelTrain)
        model = models.Sequential()
        model.add(Dense(512, input_dim=d, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(C, activation='softmax'))
        model.compile(loss=loss,
                      optimizer=optimizers.SGD(lr=lr,momentum=.5),
                      metrics=['accuracy'])
        model.summary()
        model.fit(self.data, labelTrain, epochs=epoch, batch_size=batch_size, verbose=1)
        model.save("FaceDetection.model")
    def predict(self,path):
        img = cv2.imread(path, 1)
        img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(img_gray)
        for (x, y, width, height) in faces:
            if width*height>2000:
                img_face = img_gray[y:y + height, x:x + width]
                cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),5)
        img= cv2.resize(img,(256,256))
        cv2.imwrite("image_detection.jpg",img)
        img = cv2.resize(img_face, (64, 64))
        data = img.reshape(1,4096)/255
        model = models.load_model("FaceDetection.model")
        score = model.predict(data)
        return np.argmax(score)

    def addDataFromCamera(self,name):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        i=0
        while cap.isOpened():
            frame = cap.read()[1]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray)
            date_time = str(datetime.datetime.now())[:19]
            cv2.putText(frame, date_time, (10, 10), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 255), 1, cv2.LINE_4)
            for x, y, width, height in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                if cv2.waitKey(1) == ord(" "):
                    img_name = name + '_' + str(i) + '.PNG '
                    img_path = os.path.join('database', img_name)
                    cv2.imwrite(img_path, frame[y:y + height, x:x + width, :])
                    i+=1
            cv2.putText(frame, "Number of Picture: " + str(i)+'/20', (10, 30), cv2.FONT_HERSHEY_COMPLEX, .5,
                        (0, 255, 255), 1,cv2.LINE_4)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(10) == ord('b'):
                break
            if i>19:
                break
        cap.release()
        cv2.destroyAllWindows()
    def recognitionFromCamera(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        while cap.isOpened():
            frame = cap.read()[1]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray)
            date_time = str(datetime.datetime.now())[:19]
            cv2.putText(frame, date_time, (10, 10), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 255), 1, cv2.LINE_4)
            if cv2.waitKey(10) == ord(" "):
                cap.release()
                cv2.destroyAllWindows()
                for x, y, width, height in faces:
                    if width*height>4095:
                        cv2.imwrite("live_image.jpg",frame[y:y+width,x:x+width,:])
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 7)
                        frame= cv2.resize(frame,(256,256))
                        cv2.imwrite('image_detection.jpg', frame)
            for x,y,width,height in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(10) == ord('b'):
                break
        cap.release()
        cv2.destroyAllWindows()

def UserInterface():
    if __name__=="__main__":
        app= QApplication(sys.argv)
        window=myWindow()
        sys.exit(app.exec())
UserInterface()


