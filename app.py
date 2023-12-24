import cv2
from cv2 import VideoCapture
import face_recognition
import numpy as np
import csv
import os
from pathlib import Path
from datetime import date, datetime
from flask import Flask, app, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = r"Face-Recognition-Attendance-System-main2/IMAGE_FILES"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    if 'file' not in request.files:
        flash('No file part')
        return render_template('upload.html')
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return render_template('upload.html')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html')
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return render_template('upload.html')


@app.route('/index')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    IMAGE_FILES = []
    filename = []
    dir_path = r"Face-Recognition-Attendance-System-main2/IMAGE_FILES"


    for imagess in os.listdir(dir_path):
        img_path = os.path.join(dir_path, imagess)
        img_path = face_recognition.load_image_file(img_path)  # reading image and append to list
        IMAGE_FILES.append(img_path)
        filename.append(imagess.split(".", 1)[0])

    def encoding_img(IMAGE_FILES):
        encodeList = []
        for img in IMAGE_FILES:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

        
    def read_lines_after_last_occurrence(filename):
        # Read the entire file
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Find the last occurrence of "$$$"
        last_occurrence_index = -1
        for i in range(len(lines) - 1, -1, -1):
            if "$$$" in lines[i]:
                last_occurrence_index = i
                break

        # Put the cursor after the last occurrence of "$$$" and save the rest of the lines in a list
        if last_occurrence_index >= 0:
            lines_after_last_occurrence = lines[last_occurrence_index + 1:]
            return lines_after_last_occurrence
        else:
            return None
        

    def takeAttendence(name):
        try:  
            # searching in todays names
            with open('attendence.csv', 'a+') as f:
                f.seek(0)
                mypeople_list = f.readlines()
                
                last_Saved_Day = mypeople_list[-1].split(", ", 1)[1].split()[0]
                current_Day = date.today()
                last_Saved_Day_Obj = datetime.strptime(last_Saved_Day, "%Y-%m-%d").date()
                current_Date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if current_Day != last_Saved_Day_Obj:   #if new day
                    f.writelines(f'\n\n\n\t{current_Day}\t$$$')     #write current day
                    f.writelines(f'\n{name}, {current_Date}')   #write his name,date
                    return

                filename = 'attendence.csv'
                lines_after_last_occurrence = read_lines_after_last_occurrence(filename)
                flag =0

                for line in lines_after_last_occurrence:
                    if name in line:
                        print("name is already taken successfully")
                        return
                    
                f.writelines(f'\n{name}, {current_Date}')
                
        except FileNotFoundError:
            print("the 'attendance.csv' file does not exist. Creating a temporary file...")
            with open('attendance.csv', 'w+'):
                pass


    encodeListknown = encoding_img(IMAGE_FILES)
    # print(len('sucesses'))

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # converting image to RGB from BGR
        imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fasescurrent = face_recognition.face_locations(imgc)
        encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)

        # faceloc- one by one it grab one face location from fasescurrent
        # than encodeFace grab encoding from encode_fasescurrent
        # we want them all in same loop so we are using zip
        for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
            matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
            face_distence = face_recognition.face_distance(encodeListknown, encodeFace)
            # print(face_distence)
            # finding minimum distence index that will return best match
            matchindex = np.argmin(face_distence)

            if matches_face[matchindex]:
                name = filename[matchindex]     #.upper()
                y1, x2, y2, x1 = faceloc
                # multiply locations by 4 because we above we reduced our webcam input image by 0.25
                # y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                takeAttendence(name)  # taking name for attendence function above

        # cv2.imshow("campare", img)
        # cv2.waitKey(0)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
