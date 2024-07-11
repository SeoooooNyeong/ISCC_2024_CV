import cv2

# 카메라 인덱스를 설정합니다. 카메라의 개수와 순서에 맞게 수정하세요.
camera_indices = [0, 1, 2, 3, 4]

# 각 카메라의 VideoCapture 객체를 저장할 리스트를 만듭니다.
caps = []

# 모든 카메라를 엽니다.
for index in camera_indices:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"카메라 {index}를 열 수 없습니다.")
    else:
        caps.append((index, cap))

while True:
    for index, cap in caps:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Camera {index}', frame)
        else:
            print(f"카메라 {index}에서 프레임을 읽을 수 없습니다.")

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 카메라를 닫고 리소스를 해제합니다.
for index, cap in caps:
    cap.release()

cv2.destroyAllWindows()
