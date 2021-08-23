# モデルの読込
# 保存したjsonファイルとhdf5ファイルを読み込む。モデルを学習に使うにはcompileが必要。
from keras.models import model_from_json
import cv2

# JSON形式のデータを読み込んでモデルとして復元。学習で使うにはまたコンパイルが必要なので注意。
with open('mnist.model', 'r') as f:
  json_string = f.read()
model = model_from_json(json_string)

# モデルにパラメータを読み込む。前回の学習状態を引き継げる。
model.load_weights('param.hdf5')
print('Loaded the model.')

def predict_digit(filename):
  # 自分で用意した手書きの画像ファイルを読み込む
  img = cv2.imread(filename)
  plt.imshow(img)

  # 画像データを学習済みデータに合わせる
  # グレースケールに変換
  # 2値化, 白黒反転, ガウシアンフィルタで平滑化、リサイズ
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
  img = cv2.bitwise_not(img)
  img = cv2.GaussianBlur(img, (9,9), 0)
  img = cv2.resize(img,(28, 28), cv2.INTER_CUBIC) # 訓練データと同じサイズに整形

  # float32に変換して正規化
  img = img.astype('float32')
  img = np.array(img)/255

  # モデルの入力次元数に合わせてリサイズ
  img = img.reshape(1, 28, 28, 1)
  # データを予測する
  predict_y = model.predict(img)
  predict_number = np.argmax(predict_y)

  return predict_number
