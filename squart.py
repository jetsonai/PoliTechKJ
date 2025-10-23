import cv2
import math
import collections
import mediapipe as mp

# --------- 입력 소스 설정 ----------
# GStreamer (예: Jetson, /dev/video1)
gst_str = (
    "v4l2src device=/dev/video1 ! "
    "video/x-raw, width=640, height=480, format=(string)YUY2, framerate=30/1 ! "
    "videoconvert ! video/x-raw, width=640, height=480, format=BGR ! appsink"
)
movifile = "floor.mp4"

# capsrc = movifile   # 동영상 파일로 테스트하려면 이 줄을 사용
capsrc = gst_str      # 웹캠/카메라가 기본

# --------- MediaPipe 초기화 ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    #upper_body_only=False,
    smooth_landmarks=True
)
mp_drawing = mp.solutions.drawing_utils

# --------- 유틸 함수들 ----------
def to_xyv(landmarks, id_):
    lm = landmarks.landmark[id_]
    return lm.x, lm.y, lm.visibility

def calc_angle(ax, ay, bx, by, cx, cy):
    """
    각도(도): ∠ABC, B가 꼭짓점. (A-B-C)
    """
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return None
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))

def pick_side_by_visibility(landmarks):
    """
    좌/우 무릎 쪽 중 visibility 합이 더 큰 쪽을 선택.
    """
    L = mp_pose.PoseLandmark
    # Left: hip/knee/ankle
    lx, ly, lv = to_xyv(landmarks, L.LEFT_HIP), to_xyv(landmarks, L.LEFT_KNEE), to_xyv(landmarks, L.LEFT_ANKLE)
    # Right
    rx, ry, rv = to_xyv(landmarks, L.RIGHT_HIP), to_xyv(landmarks, L.RIGHT_KNEE), to_xyv(landmarks, L.RIGHT_ANKLE)

    l_vis = lx[2] + ly[2] + lv[2]
    r_vis = rx[2] + ry[2] + rv[2]
    return "left" if l_vis >= r_vis else "right"

def get_knee_angle_xy(landmarks, side, img_w, img_h):
    """
    선택된 side의 (hip–knee–ankle) 각도를 픽셀 좌표로 계산.
    """
    L = mp_pose.PoseLandmark
    if side == "left":
        hip  = landmarks.landmark[L.LEFT_HIP]
        knee = landmarks.landmark[L.LEFT_KNEE]
        ankle= landmarks.landmark[L.LEFT_ANKLE]
    else:
        hip  = landmarks.landmark[L.RIGHT_HIP]
        knee = landmarks.landmark[L.RIGHT_KNEE]
        ankle= landmarks.landmark[L.RIGHT_ANKLE]

    ax, ay = hip.x * img_w, hip.y * img_h
    bx, by = knee.x * img_w, knee.y * img_h
    cx, cy = ankle.x * img_w, ankle.y * img_h
    return calc_angle(ax, ay, bx, by, cx, cy)

# --------- 하이퍼파라미터 ----------
DOWN_THRESHOLD = 100.0   # 무릎 각도가 이 값보다 작으면 "DOWN"(하강 완료)으로 간주
UP_THRESHOLD   = 160.0   # 무릎 각도가 이 값보다 크면 "UP"(기립 완료)으로 간주
SMOOTH_N       = 5       # 이동평균 윈도우 크기
MIN_HOLD_FR    = 3       # 상태 전환 디바운스(최소 프레임 유지)

# --------- 상태 변수 ----------
angle_window = collections.deque(maxlen=SMOOTH_N)
state = "UP"             # 시작 상태를 UP으로 가정
hold_counter = 0
reps = 0

# --------- 캡처 시작 ----------
cap = cv2.VideoCapture(capsrc)

if not cap.isOpened():
    print("[ERROR] Cannot open video source.")
    exit(1)

# --------- 메인 루프 ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    cur_angle = None
    used_side = None

    #TODO 1








    # --------- 상태머신으로 스쿼트 카운트 ----------
    # 조건:
    #  - UP 상태에서 각도 <= DOWN_THRESHOLD인 프레임이 MIN_HOLD_FR 이상 지속 -> DOWN으로 전환
    #  - DOWN 상태에서 각도 >= UP_THRESHOLD인 프레임이 MIN_HOLD_FR 이상 지속 -> UP으로 전환(=1회 증가)
    #TODO 2
    
    
    

    # --------- HUD(오버레이) ----------
    # 각도/측/상태/카운트 표시
    
    #TODO 3
    

    cv2.imshow("Squat Counter (MediaPipe Pose)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------- 정리 ----------
cap.release()
cv2.destroyAllWindows()
