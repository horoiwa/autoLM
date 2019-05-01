## autoLM (Auto Linear regression Modeling)

`pip install deap joblib`


### 自動化された線形回帰モデリング

- onehot化などの前処理の自動化

- パイプライン化された多項式化や標準化操作

- 並列化された遺伝的アルゴリズムによる特徴量選択

- バギング等の発展的なモデリング手法のsklearn風実装

<br>

### Getting started

1. データセットの作成

> ボストン住宅データセットを入力としてDataSetインスタンスを作成します。
`dataset.fit(X, y)`によってデータセットの前処理パイプラインが作成されます。このパイプラインは`dataset.transform(X)`によって再利用が可能です。

<br>

```
from autoLM.dataset import DataSet
from autoLM.support import load_df, load_sample

# ボストン住宅データセットをロードします。
X, y = load_df("boston")

# ボストン住宅データセットから1行だけをロードします。
X_sample, y_sample = load_sample('boston')

dataset = DataSet(project_name="sample_project",
                  criterio=15,
                  poly=2,
                  stsc=True
                  )

# 前処理パイプラインの作成
dataset.fit(X, y)

# 変数タイプの自動検出がうまくいっているかを確認
print(dataset.fmap)

# 前処理済みデータセットへのアクセス
X_post = dataset.get_X_processed()

# 作成済みパイプラインの利用
X_sample_post = dataset.transform(X_sample)

```
Parameters:

`project_name` : Str <br>
この名前のディレクトリが作成され、ログはここに吐き出される

`criterio` : int (default=15)<br>
データセット内のユニークな特徴数がこの数以下のカラムは, 文字列の場合はカテゴリ変数、数値の場合は順序変数と見なされる。カテゴリ変数についてはone-hotエンコーディングが行われる。

`poly` : int 0, 1, 2... <br>
sklearn.preprocessing.PolynominalFeatures による交差項の追加を行うかどうか。0,1の場合は何もしない。

`stsc` : bool (default=True)<br>
標準化を行うかどうか

<br>

Methods:

`fit(X, y)` : X, y == pd.DataFrame <br>
データセットを記録し、前処理を行う. 一度しか利用できない。


`get_X_processed()` :<br>
 前処理済みのデータセットを取得 <br>

`transform(X)` :<br>
fitメソッドによって作成された前処理パイプラインに従ってデータ操作を行う<br>

2. 遺伝的アルゴリズムによる変数選択

>上で作成した`dataset`を入力として変数のスクリーニングを行います。

```
from autoLM.feature_screening import FeatureScreeningGA

screening = FeatureScreeningGA(dataset, n_features=(5, 20),
                               n_gen=50, n_eval=250)

screening.run(prescreening=30, postscreening=10, n_jobs=1)

```
FeatureScreeningGA:

Parameters:

dataset : autoLM.dataset.Dataset

n_features : tuple (min&&int, max&&int)<br>
(絞り込みたい変数の数、許容する変数の数)

n_gen : int 遺伝的アルゴリズムの世代数

n_eval : 何度繰り返しテストセットを評価するか


Methods:

run(prescreening, postscreening, n_jobs) :<br>
prescreening, postscreening :<br>
>遺伝的アルゴリズムによって変数の数をn_features[0]個まで絞り込むという操作をprescreening回繰り返します。
その後,　prescreeningで1回以上選ばれたすべての変数から変数の数をn_features[0]個まで絞り込むという操作をpostscreening回繰り返します。結果はすべてログ(`project_name/FeatureScreeningGA/summary.txt`など)に吐き出されます。

n_jobs : 使用するスレッド数の指定

<br>

3. RidgeRPRS

> Ridge with Random Patches and Random Subspaces

```
    X, y = load_df("boston")
    X_sample, y_sample = load_sample('boston')

    dataset = DataSet(project_name, poly=1)
    print(dataset)
    dataset.fit(X, y)

    model = RidgeRPRS(dataset, n_models=100)
    model.evaluate()

    model.fit()
    mean, std = model.predict(dataset.transform(X_sample))
    print("Pred:", mean, std)
    print("Obs:", y_sample)

```
