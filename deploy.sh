#!/bin/zsh
# 切换至工作目录
echo "切换至工作目录"
cd /root/AI-RPA/

echo "更新项目"
git pull

echo "结束element_recognition_main进程"
# 找到名为element_recognition_main.py的Python进程ID
PID=$(ps -ef | grep element_recognition_main.py | grep -v grep | awk '{print $2}')
# 如果找到了进程，杀死它
if [ -n "$PID" ]; then
  kill $PID
  echo "结束element_recognition_main进程"
fi

echo "启动element_recognition_main"
cd /root/AI-RPA/
nohup python3 element_recognition/element_recognition_main.py &

echo "重新部署完成"




