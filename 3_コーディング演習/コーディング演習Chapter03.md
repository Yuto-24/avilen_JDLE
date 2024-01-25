# 3. MLP基礎（バッチ編）

## 概要
本演習ではChapter03で学習した、深層学習の基礎である多層パーセプトロン(バッチ処理)を穴埋め形式で実装します。<br>
予め用意されたコードはそのまま使用し、指示された穴埋め部を編集してください。
演習問題文は<font color="Red">赤字</font>で表示されています。<br>
また、乱数設定により実行結果が異なるため、<font color="Red">コードを完成させたあと、必ずもう一度一番上のセルから順に最後まで実行して結果を確認してください。</font>

所要時間：5~8時間

今回もデータセットとしてMNISTを使用し、`sklearn`の`train_test_split`を用いて訓練データ:テストデータ = 8:2 に分割します。

## 最適化・バッチ正規化・ミニバッチ学習（スクラッチ）

### データのサンプリング
画像・ラベルデータをランダムにいくつか取り出して可視化します。
画像は784要素の1次元ベクトルとしてXに格納されていますが、画像として表示するときは28x28の二次元にreshapeします。

### Optimizerの実装

* <font color="Red">問1. 確率的勾配降下法を用いたOptimizerのクラス SGD を完成させてください。</font><br>
SGDの特徴として、データを変えることによって、損失関数自体を確率的に変えることができることが挙げられます。これによって、パラメータは同じであっても、勾配の向きを確率的に換えることができます。
この問題ではSGDのコードを回答してください。

    ```python
    class SGD:

        def __init__(self, lr=0.01):
            self.lr = lr

        def update(self, params, grads):
            for key in params.keys():
                params[key] -= self.lr * grads  # 問1 ### 旧シラバスコース演習3問4と同じ

    ```

* <font color="Red">問2. Adamクラス を完成させてください。</font><br>
Adamの特徴としてハイパーパラメータのバイアス補正(偏りの補正)が行われることが挙げられます。書籍『ゼロから作るDeepLearning』の配布コードは簡易版のため、バイアス補正を組み込んでいません。<br>
この問題ではバイアス補正を組み込んだ完成形のAdamコードについて回答してください。また、過去のE資格試験ではこちらの完成形のAdamコードが出題されています。

#### 講義の復習

まず、Adamの式について考えてみます。Adamの式は、講義スライドに載っているので合わせてご覧ください。

Adamではまず、`m`と`v`という２つの変数の更新を行います。

`m`の更新については、ハイパーパラメータ`beta1`と勾配値`grads`を用いると次のように表現することができます。

$$ m = \text{beta1} \cdot m + (1 - \text{beta1}) \cdot \text{grad} $$

上記と同じように、`v`の更新については、ハイパーパラメータ`beta2`と勾配値`grads`を用いると次のように表現することができます。





$$ v = \text{beta2} * v + (1 - \text{beta2}) * \text{grad}^2 $$

次に、`m`と`v`のバイアスの補正について説明します。

ここでは`m`のバイアス補正を`m-unbias`とします。`m`とハイパーパラメータ`iter`と`beta1`で表すと以下の通りになります。

$$ \text{m-unbias} = \frac{m}{1 - \text{beta1}^\text{iter}} $$

上記と同じようにして、`v`のバイアス補正を`v-unbias`とします。`v`とハイパーパラメータ`iter`と`beta2`で表すと以下の通りになります。

$$ \text{v-unbias} = \frac{v}{1 - \text{beta2}^\text{iter}} $$

最後に、パラメータ`params`の更新を行います。

これまで求めたものとハイパーパラメータ`lr`を用いて、パラメータの更新を行うと以下の通りになります。

$$ \text{params} = \text{params} - \text{lr} * \frac{\text{m-unbias}}{\sqrt{\text{v-unbias}} + ϵ} $$

なお、$ϵ$についてですが、$\text{v-unbias} = 0$のとき、分母が0になることでゼロ除算することを防ぐために加えます。$ϵ$は微小値（10^(-7)など）で構いません。

これらをコードに表現することができれば完成です。書いてみましょう。

#### コスト関数

多クラス分類問題なので、クロスエントロピーをコスト関数として用います。

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

```

### バッチ正規化を用いないMLP

まずはバッチ正規化を入れない普通の三層ニューラルネットワークを実装します。問題にはなっていませんが、chapter02の復習も兼ねてコードを読み理解しておいてください。

### バッチ正規化を用いたMLP
* <font color="Red">問3. バッチ正規化を用いたMLPクラスを完成させる</font><br>
    * <font color="Red">問3-1. 【forward関数】ミニバッチの平均を算出する処理を記述してください。</font><br>
    * <font color="Red">問3-2. 【forward関数】ミニバッチの分散を算出する処理を記述してください。</font><br>
    * <font color="Red">問3-3. 【forward関数】移動平均により全体の平均を求める処理を記述してください。</font><br>
    * <font color="Red">問3-4. 【forward関数】移動平均により全体の分散を求める処理を記述してください。</font><br>
    * <font color="Red">問3-5,3-6. 【forward関数】テスト時における全体の平均と分散を使った正規化する処理を記述してください。</font><br>
    * <font color="Red">問3-7,3-8. 【backward関数】β, γの勾配を算出する処理を記述してください。</font><br>

各層について、重みを掛けて足し合わせた後バッチ正規化を行う。

バッチ正規化の順伝播は以下の式に従って実装します。

  - （訓練時のみ）まずは計算しているミニバッチについて、平均と分散を求めます。各次元について、全データを通じた平均・分散を計算するため、平均・分散を計算する軸にご注意ください。

  - （訓練時のみ）テスト時に使用するために、訓練データ全体での平均を推定します。モーメンタム $m$ を用いて今までの平均 $\mu_{old} $ を計算しているミニバッチの平均 $\mu$ の方向に移動させ、新しい平均$\mu_{new} $を求めます。
  $$
    \mu_{new} = m \mu_{old} + ( 1 - m)\mu\tag{1}
  $$

  - （訓練時のみ）同様に今までの分散 $\sigma_{old} ^ 2$ を計算しているミニバッチの平均 $\sigma^2$の方向に移動させ、 新しい分散$\sigma_{new}^2$ を求めます。
  $$
    \sigma_{new}^2 = m \sigma_{old}^2 + ( 1 - m)\sigma^2\tag{2}
  $$

  - 求めた平均 $\mu$ と分散 $\sigma^2$ を用いて、入力 $x$ を正規化した値 $x_n$ を求めます。分散$\sigma^2$から標準偏差 $\sigma$ を求めるときに、アンダーフローを避けるために 1e-6 ($10 ^ {-6}$) を足してから平方根を取っています。
  テスト時には、移動平均により推定した訓練データ全体での平均・分散を使用します。
  $$
    \sigma = \sqrt{\sigma ^ 2 + 10 ^ {-6}}\tag{3}
  $$
  $$
    x_n = (x - \mu) / \sigma\tag{4}
  $$

  - 正規化した値 $x_n$に対して $\gamma$ を用いて変倍し、$\beta$ を用いて移動を行い、活性化関数に渡す出力 $y$ を求めます。
  $$
    y = \gamma x_n + \beta\tag{5}
  $$

バッチ正規化の誤差逆伝播は以下の式に従って実装します。

  - 直前まで逆伝播してきた$1, 2, \dots , N$ 番目(Nはバッチサイズ)の出力データ$y_k$による勾配 $\frac{\partial L}{\partial y_k}$を用いて $\gamma$ と$\beta$による勾配を計算します。 $x_{nk}$ はミニバッチの中のk番目の入力データを正規化した後の値を表します。
  $$
    \begin{alignat}{99}
      \frac{\partial L}{\partial \gamma} & = & \sum_{k=1}^{N} \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial \gamma} = \sum_{k=1}^{N} \frac{\partial L}{\partial y_k} x_{nk}\tag{6} \\
      \frac{\partial L}{\partial \beta} & = & \sum_{k=1}^{N} \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial \beta} = \sum_{k=1}^{N} \frac{\partial L}{\partial y_k}\tag{7}
    \end{alignat}
  $$

  - $1, 2, \dots , N$ 番目の入力データ$x_k$による勾配 $\frac{\partial L}{\partial x_k}$を計算します（コードでは高速化のため少々異なった計算をしています）。

  $$
    \begin{equation}
      \frac{\partial L}{\partial x_k} = \frac{\gamma}{\sigma} \Bigg[ \frac{\partial L}{\partial y_k} - \frac{1}{N} \bigg[ \frac{\partial L}{\partial \beta} + x_{nk} \frac{\partial L}{\partial \gamma} \bigg] \Bigg]
    \end{equation}
  $$

### ミニバッチを用いた学習

ミニバッチを用いた学習を行います。

* <font color="Red">問4. ランダムサンプリングのためにインデックスをランダムに読み込む処理を記述しましょう。</font>
* <font color="Red">問5. バッチサイズごとに入力と教師データを読み込む処理を記述しましょう。</font>

ヒント: numpy.random.permutation を用いることで、データのインデックスをシャッフルした配列を用意することで、シャッフルインデックス配列permに対して、前からバッチサイズずつインデックスを切り出せばミニバッチの抽出が行えます。

また、学習用のコードは実行に時間がかかります。完了するまで5~10分ほどを要しますのでご注意ください。

#### <font color="Red">問3-3/3-4　ヒント</font>

次に、移動平均による全体の平均・分散について考えます。

移動平均による全体の平均については、解説文の式（１）、分散の平均については、解説文の式（２）に書かれております。

まず、平均については次の通りに求めることができます。

$$ \mu_{new} = m \mu_{old} + ( 1 - m)\mu $$

ここで、`m`とは、`self.momentum`、`μ_old`は`self.norms['mu' + str(idx)]`、`μ`は`mu`を表します。

次に、分散については次の通りに求めることができます。

$$ \sigma_{new}^2 = m \sigma_{old}^2 + ( 1 - m)\sigma^2 $$

ここで、`m`とは、`self.momentum`、`σ_old`は`self.norms['var' + str(idx)]`、`σ`は`var`を表します。

使用する変数と計算方法が分かれば、コーディングできるはずです。



