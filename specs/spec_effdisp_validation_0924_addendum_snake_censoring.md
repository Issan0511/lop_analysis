# 0924 追補2 — Snakeと、平衡前の応答消失

本走前の追加。Issa「他の活性化は先に勾配が尽きるかも」「Snakeも入れて」。元spec290c4f8、追補2ed7321を保持する。

## Snake

固定α=1のSnake、phi(z)=z+sin(z)^2、phi'(z)=1+sin(2z)。適応αは今回含めない。
CondA基準設定r5/f15/k1/Adamにk1_snakeを追加し計14腕。RL-MNIST/PMにもSN1を追加し各6活性化×5seed、他の設定は既登録と同じ。
Codex予測: 負側への輸送によって微分が一様に消えることはないため、ReLU/ELUよりLOW_RESPONSEは少ない。ただし釣り合い・頭打ち・予測精度の成立は保証しない。leakyとSnakeでcが等しいとは予測しない。

## 応答消失で打ち切られる場合

一次判定は元specの全地平P1–P6をそのまま残す。低応答による打ち切りを併記する。
LOW_RESPONSE onset: いずれかの隠れ層で、task末の入力上 derivative_absmean<1e-8 またはactiveunit_frac<.1が3task連続したとき、その3taskの最初をonsetとする。CondAは1層、MNISTは2層。
これは計測上の低応答ラベルで、一般的なLoPの定義や因果判定ではない。
条件付きのreportとして、onsetより前の最後20transitionsを解析する。20本未満ならINSUFFICIENT_ACTIVE_HISTORY。onsetなしはこの追加窓なし。
ここでの20本窓の収支・成長率を記録するが、元のfull-horizon登録判定を置き換えない。停止後のplateauをactive equilibriumの支持にしない。平衡に達する前に停止した走では、active equilibriumが成立しないと断定せず「到達前に打ち切り」と書く。

MNISTは新課題の学習前accuracyと、決定的な最初16例の生勾配norm(W1/W2)を開始/終了時に保存する。学習完了で当該課題の勾配が小さくなることと、新課題へ応答できないことを区別する補助量。CondAも新課題の学習前の全supportMSEを保存する。

ELU実装は今回のGPU比較に合わせてPyTorch F.elu(alpha=1,inplace=False)、負側derivative=exp(z)。float32の負側tailをautogradと照合する。古いexpm1を使った実験のoutput+1床とは同一条件ではない。ReLU/GELU/SiLU/Snakeも実際のforwardに対応する微分を使う。
