import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from element_recognition.element_recognition_server import app
import element_recognition.controller

if __name__ == '__main__':
    app.run(port=10088, host='0.0.0.0')