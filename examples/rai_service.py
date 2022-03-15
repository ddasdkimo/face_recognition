# 这是一个非常简单的使用Web服务上传图片运行人脸识别的案例，后端服务器会识别这张图片是不是奥巴马，并把识别结果以json键值对输出
# 比如：运行以下代码
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
# 会返回：
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# 本项目基于Flask框架的案例 http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# 提示：运行本案例需要安装Flask，你可以用下面的代码安装Flask
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect
import cv2
import numpy
import os
import time
# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # 检测图片是否上传成功
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        file1 = request.files['file1']
        if file.filename == '':
            return redirect(request.url)
        if file1.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename) and file1 and allowed_file(file1.filename):
            # 图片上传成功，检测图片中的人脸
            return detect_faces_in_image(file,file1)

    # 图片上传失败，输出以下html代码
    return '''
    <!doctype html>
    <title>臉部比對</title>
    <h1>上傳兩張圖片比較是否為同一個人</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="file" name="file1">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file_stream,file_stream1):

    # 载入用户上传的图片
    # img = face_recognition.load_image_file(file_stream)
    if not os.path.isdir("tmpdata"):
        os.mkdir("tmpdata")
    img1 = face_recognition.load_image_file(file_stream)
    img2 = face_recognition.load_image_file(file_stream1)
    unknown_face_encodings = face_recognition.face_encodings(img1)
    unknown_face_encodings1 = face_recognition.face_encodings(img2)
    t = time.time()
    cv2.imwrite("tmpdata/"+str(t)+"_1.jpg",cv2.cvtColor(numpy.asarray(img1),cv2.COLOR_RGB2BGR))
    cv2.imwrite("tmpdata/"+str(t)+"_0.jpg",cv2.cvtColor(numpy.asarray(img2),cv2.COLOR_RGB2BGR))

    face_found = False
    is_same_person = False

    if len(unknown_face_encodings) > 0 and len(unknown_face_encodings1):
        face_found = True
        # 看看图片中的第一张脸是不是相同
        match_results = face_recognition.compare_faces([unknown_face_encodings1[0]], unknown_face_encodings[0],tolerance=0.4)
        if match_results[0]:
            is_same_person = True

    # 讲识别结果以json键值对的数据结构输出
    result = {
        "file1_face_len": len(unknown_face_encodings),
        "file2_face_len": len(unknown_face_encodings1),
        "is_same_person": is_same_person
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
