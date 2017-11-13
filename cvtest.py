import time
import pyautogui

for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)

while True:
    print('down')
    pyautogui.keyDown('s')
    time.sleep(3)
    print('up')
    pyautogui.keyDown('w')
    time.sleep(3)
