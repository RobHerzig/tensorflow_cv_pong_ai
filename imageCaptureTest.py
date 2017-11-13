import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D

#vars -----------------------------------------------------
player_left_pos = (0,0) #use these positions and the distance player->ball as input for NN
player_right_pos = (0,0)
ball_pos = (0,0)

templatePlayer = cv2.imread('leftPlayer.png', 0)
templateBall = cv2.imread('ball.png', 0)
#templates for the score
templateScore0 = cv2.imread('template_zero.png', 0)
templateScore1 = cv2.imread('template_one.png', 0)
templateScore2 = cv2.imread('template_two.png', 0)
templateScore3 = cv2.imread('template_three.png', 0)
templateScore4 = cv2.imread('template_four.png', 0)
templateScore5 = cv2.imread('template_five.png', 0)

relevant_roi_vertices = np.array([[92, 638], [92, 70], [1188, 70], [1188, 638]])
vertices_score = np.array([[560, 740], [560, 680], [720, 680], [720, 740]])
#cap = cv2.VideoCapture(0)  # only for testing
last_time = time.time()

widthPlayer, heightPlayer = templatePlayer.shape[::-1]
widthBall, heightBall = templateBall.shape[::-1]

#endvars -----------------------------------------------------------

# define region of interest inbetween vertices
def roi(img, vertices):
    mask = np.zeros_like(img)
    #print (vertices)
    cv2.fillPoly(mask, np.int32([vertices]), 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

#Let neural network simply output [a,b] where a/b=0 -> release/don't press W/S, a/b=1 -> don't release/press W/S
def goUp():
    PressKey(W)

def releaseUp():
    ReleaseKey(W)

def goDown():
    PressKey(S)

def releaseDown():
    ReleaseKey(S)

#for testing controls (simply presses W and S via DirectX input)
def stupidTest(times):
    counter = 0
    while counter < times:
        counter += 1
        print(counter)
        print('down')
        PressKey(S)
        time.sleep(2)
        ReleaseKey(S)
        print('up')
        PressKey(W)
        time.sleep(2)
        ReleaseKey(W)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 2)
    except:
        pass


def process_img(original_image):
    # processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(original_image, threshold1=80, threshold2=120)
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    vertices = relevant_roi_vertices
    #vertices = vertices_score
    processed_img = roi(processed_img, [vertices])

    #                       Edges! in this case: canny
    #lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 90, np.array([]), 35, 3)
    #draw_lines(processed_img, lines)
    return processed_img

def read_score(image):
    print("Read Score")

#make NN minimize this -> should always try to hit the ball
def get_ball_player_distance(pos1, pos2):
    return abs(pos1[1] - pos2[1])

def find_players(image):
    #cv2.imshow('test_window', image)
    #cv2.waitKey(0)
    image = roi(image, relevant_roi_vertices)
    #Find Players
    resPlayer = cv2.matchTemplate(image, templatePlayer, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    locPlayers = np.where(resPlayer >= threshold)

    #Find Ball
    resBall = cv2.matchTemplate(image, templateBall, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    locBall = np.where(resBall >= threshold)
    try:
        player_left_pos = tuple(x[0] for x in locPlayers)
        ball_pos = tuple(x[0] for x in locBall)
        print("IN METHOD: PlayerPosition " + str(player_left_pos) + " BallPosition " + str(ball_pos))

        x_values = [player_left_pos[0], ball_pos[0]]
        y_values = [player_left_pos[1], ball_pos[1]]

        result = (player_left_pos, ball_pos)
        #result = zip(x_values, y_values)
        return result
    except:
        return ()

full_width = 1280
full_height = 740
counter = 3
while counter >= 0:
    counter -= 1
    time.sleep(1)
    print(counter+1)
while True:
    screen = np.array(ImageGrab.grab(bbox=(0, 40, full_width, full_height)))
    #screen = cv2.resize(screen, (int(full_width/2), int(full_height/2)))
    #ret, screen = cap.read()

    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('BlackWhite', screen)
    positions = find_players(screen)

    new_screen = process_img(screen)

    try:
        print ("Positions" + str(positions))
        player_left_pos = (positions[0][1],positions[0][0])
        ball_pos = (positions[1][1],positions[1][0])
        print("PlayerPosition " + str(player_left_pos) + " BallPosition " + str(ball_pos))
        print("DISTANCE: " + str(get_ball_player_distance(player_left_pos, ball_pos)))
        cv2.rectangle(new_screen, player_left_pos, (player_left_pos[0] + widthPlayer, player_left_pos[1] + heightPlayer), (255, 255, 255), 1)
        cv2.rectangle(new_screen, ball_pos, (ball_pos[0] + widthBall, ball_pos[1] + heightBall), (255, 255, 255), 1)
    except:
        pass
    # #printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8')\
    # 	#.reshape((printscreen_pil.size[1], printscreen_pil.size[0], 3))
    print('Loop took {} seconds'.format(time.time() - last_time))
    last_time = time.time()

    cv2.imshow('windowProcessed', new_screen)

    # cv2.imshow('Window', cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
