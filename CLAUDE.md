# vrc-ozaki-schema-LT

## プロジェクト概要

GeForce RTX (RTX 5080, Blackwell sm_120) で、CUDA 13.4 の cuBLAS FP64 エミュレーション
(Ozaki scheme I / II) による DGEMM・PDE 計算の高速化と精度を、Docker で分離した
CUDA 12 環境との対照実験で定量評価する。成果は VRChat 物理学集会の LT で発表する (slides-jp/)。

### 技術仕様

- 条件: `cuda12` (12.9.1) / `cuda13-native` / `cuda13-emu` (performant) / `cuda13-emu-eager`。
  エミュレーションは `CUBLAS_EMULATE_DOUBLE_PRECISION` で実行時に切り替える
- CPU ベースライン: `cpu-eigen` (C++17 + Eigen 3.4 ネイティブ GEMM + OpenMP, `cpp/cpu_bench`) と
  `cpu-numpy` (OpenBLAS)。入力は Python が `.npy` で渡し、精度評価も Python 側の同じロジックで行う。
  スレッド 6 / 14 / 16 (P / P+E / 全コア) を掃引し、最速を代表値にする
- Fortran drop-in (`fortran/`, `fortran` サービス = cuda13 + gfortran + OpenBLAS): OpenBLAS にリンクした
  同一バイナリを `LD_PRELOAD=libnvblas.so.13` で cuBLAS に回し、エミュレーションを環境変数で切り替える。
  時間は PCIe 転送込み。NVBLAS は失敗しても `dgemm_` が黙って戻るので、ランナーが毎回 NVBLAS のログを確認する
- 両イメージともホストドライバの libcuda を使う (cuda13 イメージから forward-compat libcuda を削除済み)
- ワークロード: DGEMM、2D 熱方程式 (ADI-GEMM / FTCS ステンシル = 負の対照)
- 精度: NumPy 比較、厳密丸め参照 (TwoProduct + fsum) とのサンプル比較、PDE の離散厳密解との比較
- カーネル確認: `profile_kernels.sh` (nsys) で実際に動いたカーネルから Ozaki-I / II / native を判定

### 進捗

- [x] ベンチマーク環境 (Docker / スクリプト / レポート生成)
- [x] 本番測定 (phi=0.5): DGEMM 最大 19.2x、ADI 最大 15.9x、ステンシル 1.00x
- [x] nsys によるカーネル確認、phi 掃引
- [x] C++/Eigen の CPU ベースライン: Eigen 比で GPU native 2.5x、GPU Ozaki 48x (DGEMM N=16000)、ADI 39x
- [x] Fortran + NVBLAS の drop-in: 再ビルドなしで Ozaki-II が動く。OpenBLAS 比 10.4x (N=16000)。
  PCIe x4 のため転送が約 7 割。スライドでは Appendix に置いた
- [ ] スライド作成: ドラフト (本編 20 枚 + Appendix 7 枚) を `feature/slides-draft` で作成中。
  表紙のイラスト、cuSOLVER の数値の取り直し、スライド用の図は未着手

## 言語設定

このプロジェクトでは**日本語**での応答を行ってください。ただし、コード内のコメント、ログメッセージ、エラーメッセージ、ドキュメンテーション文字列などは**英語**で記述してください。

## 開発ルール

### コーディング規約

- Python: PEP 8準拠
- 関数名: snake_case
- クラス名: PascalCase
- 定数: UPPER_SNAKE_CASE
- Docstring: Google Style

### スライド作成

[lt-slide-template](https://github.com/s-sasaki-earthsea-wizard/lt-slide-template) (commit `45cd3ad`) の規約に従う。

- ベース: reveal.js 5.1.0。メインファイルは `slides-jp/index.html` (Markdown を `<textarea>` に書く)
- スタイル: `slides-jp/styles/custom-style.css` (「水彩の地図帳」テーマ。sky テーマの変数を上書きしている)。
  テンプレートから変えるときは、テンプレート側にも戻すかを考える
- **1280×720 (16:9)。** 各スライドは flex の縦並びで、見出しが上、本文はフッターとのあいだで縦に中央寄せ。
  本文はフッター (上端 652 px) より上に収める。はみ出したら文を削るか図を小さくする
- **章とフッター。** 章の最初のスライドに `<!-- .slide: data-chapter="章名" -->` を書くと、次の章までがその章になる。
  表紙は `class="title-slide"`、Appendix の扉は `class="divider" data-appendix="true"` (ここから先は章名だけ出す)
- **表紙のイラスト** は `slides-jp/assets/images/title.png` を 16:9 の絵に差し替える (今は白一色のプレースホルダー)。下 192 px は題の帯で隠れる
- 手元での表示は `make slides` (http://localhost:8000、`SLIDES_PORT` / `SLIDES_BIND` で変更)
- フォントは `slides-jp/vendor/fonts` に同梱したサブセット。**スライドを書いたら `make fonts` を流す** (`scripts/subset-fonts.sh`)
- スライドの区切りは「`---` だけの行」。発表者ノートは `Note:` の行から下 (`note:` / `notes:` という文字列を本文に書かない)
- 数式は KaTeX (`$...$`)。`\{`、`\,`、`\|` を使わない。下付き `_` を 2 つ以上含む数式は空行を挟まない `<div>` で囲む。
  上付きの数字 (`10⁻¹³` など) はフォントにないので KaTeX で書く (`$10^{-13}$`)
- 太字の中身が記号で始まるときは `**` ではなく `<b>`。HTML ブロック (表など) の中は Markdown が効かないので `<b>` を使う
- 部品: `simple-box`、`highlight-box`、`results-box` (表)、`summary-list`、`caption`、`credit`、`ref-list`。
  文字を詰めるときは `dense`。表は HTML の `<table>`。図と文を並べるときは `two-column` + `column-figure` / `column-text`
- インライン SVG の図は `svg.diagram`、ラベルは表示上 24 px 以上。配色は `--atlas-*` の値 (CPU = ink-muted、GPU FP64 = blue、Ozaki = orange)
- **数値はすべて `results*/` の JSON / report から取り、発表者ノートにどの実行の値かを書く**
- `.progress` などのクラス名を自作しない (reveal.js 自身のクラスとぶつかる)
- 収まりの確認は headless Chrome で全スライドを 1 枚ずつ表示して測る (`~/.cache/puppeteer` の Chrome + puppeteer-core)。
  見た目は全画面のスクリーンショットで見る

## Git運用

- ブランチ戦略: feature/*, fix/*, refactor/*
- コミットメッセージ: 英文を使用、動詞から始める
- PRはmainブランチへ

## 開発ガイドライン

### ドキュメント更新プロセス

機能追加やPhase完了時には、以下のドキュメントを同期更新する：

1. **CLAUDE.md**: プロジェクト全体状況、Phase完了記録、技術仕様
2. **README.md**: ユーザー向け機能概要、実装状況、使用方法
3. **Makefile**: コマンドヘルプテキスト（## コメント）の更新
4. **makefiles/**: コマンドヘルプテキスト（## コメント）の更新

### コミットメッセージ規約

#### コミット粒度

- **1コミット = 1つの主要な変更**: 複数の独立した機能や修正を1つのコミットにまとめない
- **論理的な単位でコミット**: 関連する変更は1つのコミットにまとめる
- **段階的コミット**: 大きな変更は段階的に分割してコミット

#### プレフィックスと絵文字

- ✨ feat: 新機能
- 🐞 fix: バグ修正
- 📚 docs: ドキュメント
- 🎨 style: コードスタイル修正
- 🛠️ refactor: リファクタリング
- ⚡ perf: パフォーマンス改善
- ✅ test: テスト追加・修正
- 🏗️ chore: ビルド・補助ツール
- 🚀 deploy: デプロイ
- 🔒 security: セキュリティ修正
- 📝 update: 更新・改善
- 🗑️ remove: 削除

**重要**: AI の支援を受けた成果物には、**AI の支援を受けたことだけ**を書く。製品名やモデル名は書かない (すぐ変わるため)。

- コミットメッセージの末尾に、次の trailer を入れる

  ```text
  Assisted-by: AI
  ```

- PR 本文の末尾に、次の 1 行を入れる

  ```text
  This PR was developed with AI assistance.
  ```
