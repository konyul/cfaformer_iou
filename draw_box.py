import cv2

# 이미지 로드
image = cv2.imread('projected_pts_img_0.jpg')

# 박스의 좌상단 좌표 (x, y)
x, y = 100, 100  # 예시 좌표, 실제로는 원하는 위치로 조정해야 합니다

# 박스 그리기 (녹색)
cv2.rectangle(image, (x, y), (x+16*15, y+16*15), (0, 255, 0), 2)

# 결과 저장
cv2.imwrite('image_with_box.jpg', image)