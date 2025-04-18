from tensorflow import keras
from sklearn.model_selection import train_test_split

import numpy as np

# Fashion MNIST 데이터셋을 로드합니다.
# Fashion MNIST는 28x28 크기의 흑백 이미지로 구성된 데이터셋입니다.
# 각 이미지는 10개의 의류 카테고리 중 하나에 속합니다.
# 데이터셋은 훈련 세트와 테스트 세트로 나뉘어 있습니다.
# 훈련 세트는 60,000개의 이미지로 구성되어 있으며, 테스트 세트는 10,000개의 이미지로 구성되어 있습니다.
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 훈련 세트와 테스트 세트의 크기를 출력합니다.
print(f'훈련 세트 크기: {train_input.shape}, {train_target.shape}')
print(f'테스트 세트 크기: {test_input.shape}, {test_target.shape}')

# 학습 데이터를 4차원 텐서(배열)로 변환합니다. 각 이미지는 28x28 크기에 채널 1(흑백)로 구성되며, 픽셀 값을 0과 1사이로 정규화합니다.
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

# 학습 데이터를 추가로 학습/검증 데이터셋으로 분할하기 위해 train_test_split 함수를 사용합니다.
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 순차 모델(Sequential model)을 생성합니다. 모델에 레이어를 순서대로 추가할 예정입니다.
model = keras.Sequential()
# 모델에 레이어를 추가합니다.
# Conv2D 레이어는 2D 컨볼루션 레이어로, 이미지에서 특징을 추출하는 데 사용됩니다.
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# MaxPooling2D 레이어는 풀링 레이어로, 이미지의 크기를 줄이고 특징을 강조하는 데 사용됩니다.
model.add(keras.layers.MaxPooling2D(pool_size=2))
# Dropout 레이어는 과적합을 방지하기 위해 일부 뉴런을 무작위로 비활성화하는 데 사용됩니다.
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
# MaxPooling2D 레이어를 추가하여 이미지의 크기를 줄입니다.
model.add(keras.layers.Flatten())
# Flatten 레이어는 다차원 배열을 1차원 배열로 변환합니다.
model.add(keras.layers.Dense(100, activation='relu'))
# Dropout 레이어를 추가하여 과적합을 방지합니다.
model.add(keras.layers.Dropout(0.4))
# 마지막 레이어는 10개의 클래스를 분류하는 Dense 레이어입니다.
model.add(keras.layers.Dense(10, activation='softmax'))
# 모델의 구조를 출력합니다.
model.summary()

keras.utils.plot_model(model)
# 모델 구조를 보강된 정보(각 레이어의 입력/출력 형태 포함)와 함께 'cnn-architecture.png' 파일로 저장합니다.
keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)

# 모델을 컴파일합니다. Adam 옵티마이저를 사용하고, 손실 함수로 sparse_categorical_crossentropy를 사용합니다.
# sparse_categorical_crossentropy는 다중 클래스 분류 문제에서 사용되는 손실 함수입니다.
# metrics 인자로 정확도를 사용합니다.
# metrics는 모델의 성능을 평가하는 데 사용되는 지표입니다.
# 모델을 컴파일한 후, 학습을 시작합니다.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습을 위한 콜백을 설정합니다.
# ModelCheckpoint 콜백은 모델의 가중치를 저장하는 데 사용됩니다.
# save_best_only=True로 설정하면, 가장 좋은 성능을 보인 모델만 저장됩니다.
# EarlyStopping 콜백은 학습이 더 이상 개선되지 않을 때 학습을 조기 종료하는 데 사용됩니다.
# patience=2로 설정하면, 2 에폭 동안 개선되지 않으면 학습을 종료합니다.
# restore_best_weights=True로 설정하면, 학습이 종료된 후 가장 좋은 성능을 보인 모델의 가중치를 복원합니다.
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

# 모델 학습을 시작합니다. epochs=20으로 설정하여 20 에폭 동안 학습합니다.
# validation_data 인자로 검증 데이터를 제공하여, 각 에폭마다 검증 성능을 평가합니다.
# callbacks 인자로 설정한 콜백을 사용하여 모델을 학습합니다.
# fit 메서드는 모델을 학습하는 데 사용됩니다. 훈련 데이터와 타겟을 인자로 받습니다.
# epochs 인자로 학습할 에폭 수를 설정합니다.
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

# 학습이 완료된 후, 모델을 평가합니다. val_scaled와 val_target을 인자로 사용하여 검증 성능을 평가합니다.
# evaluate 메서드는 모델의 성능을 평가하는 데 사용됩니다. 검증 데이터와 타겟을 인자로 받습니다.
# metrics 인자로 설정한 정확도를 사용하여 모델의 성능을 평가합니다.
model.evaluate(val_scaled, val_target)

# 모델을 사용하여 예측을 수행합니다. val_scaled의 첫 번째 이미지를 사용하여 예측합니다.
preds = model.predict(val_scaled[0:1])

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f'Predicted class: {classes[np.argmax(preds)]}')

# 모델을 사용하여 테스트 세트에 대한 성능을 평가합니다.
# test_scaled는 테스트 데이터를 4차원 텐서로 변환하고 정규화합니다.
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
# evaluate 메서드를 사용하여 테스트 세트에 대한 성능을 평가합니다.
model.evaluate(test_scaled, test_target)