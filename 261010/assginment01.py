import numpy as np
np_arr = np.array([1, 2, 3, 4, 5])

print("Sum:", np_arr.sum())
print("Minimum:", np_arr.min())
print("Maximum:", np_arr.max())
print("Mean:", np_arr.mean())
print("Argmax:", np_arr.argmax())

# [row: 학생, column: 과목]
scores = np.array([
    [83, 92, 100],  # 0번 학생
    [76, 63, 58],   # 1번 학생
    [91, 99, 84]    # 2번 학생
])

student_means = np.mean(scores, axis=1)
print("학생별 평균 (axis=1):")
print(student_means)

print()

subject_means = np.mean(scores, axis=0)
print("과목별 평균 (axis=0):")
print(subject_means)

scores = np.array([85, 90, 100, 60, 90, 75, 90])

is_ninety = (scores == 90)
print("90점인 점수 Boolean 배열:", is_ninety)
idx_ninety = np.where(scores == 90)
print(list(idx_ninety[0]))

print("90점인 학생 수:", (scores == 90).sum())
print("80점 이상인 학생 수:", (scores >= 80).sum())

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화

np_arr_normalized = (np_arr - np.min(np_arr)) / (np.max(np_arr) - np.min(np_arr))
print(np_arr_normalized)

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

np_arr_final = ((np_arr - np.min(np_arr)) / (np.max(np_arr) - np.min(np_arr))).astype(np.float32)
print(np_arr_final)
print(np_arr_final.dtype)

# < Section Project >
# 본 섹션에서 배운 내용을 토대로 프로젝트를 진행합니다.<br>
# 학교 전산 시스템에 5명 학생의 정보와 성적이 저장되어 있습니다. 학생 명단은 `"studentID_name"` 형식의 1차원 문자열 리스트로 저장되어 있으며, 성적은 5명 학생의 3개 과목 점수가 2차원 배열로 관리됩니다.<br>
# 제공된 데이터를 활용하여 아래 분석을 순서대로 수행해봅시다.

raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
    ]

raw_scores = np.array([
    [95, 82, 88],   # 짱구
    [70, 65, 90],   # 철수
    [98, 100, 92],  # 맹구
    [60, 75, 80],   # 유리
    [81, 88, 84],   # 훈이
])

subjects = ["국어", "수학", "과학"]

# TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성
student_names = [student.split("_")[1] for student in raw_students]
print("1. 학생 명단: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)

# TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 성적: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)

# TODO: 전체 12개 성적 중 85점 이상 개수
over_85 = (all_scores >= 85).sum()
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

# TODO: 학생별 평균 점수 출력
student_means = raw_scores.mean(axis=1)
print("4. 학생별 평균 점수: " + ", ".join(f"{name} {mean:.2f}점"for name, mean in zip(student_names, student_means)))  
# 예상 출력: "4. 학생별 평균 점수: 짱구 00점, 철수 00점, 맹구 00점, 유리 00점, 훈이 00점"

# TODO: 평균 점수가 가장 높은 학생 선택
top_student_idx = np.argmax(student_means)
top_student_name = student_names[top_student_idx]

print(f"5. 최고 평균 수석 학생: {top_student_name} ({student_means[top_student_idx]:.2f}점)")
# 예상 출력: "5. 최고 평균 수석 학생: XX (xx.xx점)"

# TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
overall_max_score = np.max(raw_scores)
overall_max_idx = np.argmax(raw_scores)
overall_max_subject = subjects[overall_max_idx % 3]
overall_max_student = student_names[overall_max_idx // 3]

print(f"6. 단일 과목 최고 점수: {overall_max_score}점 (과목: {overall_max_subject}, 학생: {overall_max_student})")
# 예상 출력: "6. 단일 과목 최고 점수: xx점 (과목: XX, 학생: XX)"