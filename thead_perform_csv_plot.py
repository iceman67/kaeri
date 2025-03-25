import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

# 한글 폰트 설정

# 데이터 준비
data = {
    'Thread Name': [
        'capture', 'image save', 'convert frame',
        'object detection', 'feature extraction',
        'make hash', 'sign hash', 'image send'
    ],
    'Elapsed time(ms)': [31.64, 19.12, 5.56, 76.00, 4.97, 26.10, 9.89, 11.04],
    'Std. Deviation(ms)': [6.73, 13.86, 3.50, 2.62, 2.62, 3.99, 1.76, 2.51]
}

# 그래프 그리기
df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
plt.bar(df['Thread Name'], df['Elapsed time(ms)'], yerr=df['Std. Deviation(ms)'], capsize=5)

# 그래프 타이틀 및 레이블
plt.title('Execution time and standard deviation per thread')
plt.xlabel('Thread Name')
plt.ylabel('Time(ms)')

# X축 라벨을 15도 기울이기
plt.xticks(rotation=15, ha='right')

# 레이아웃 조정 및 그래프 출력
plt.tight_layout()
plt.show()
