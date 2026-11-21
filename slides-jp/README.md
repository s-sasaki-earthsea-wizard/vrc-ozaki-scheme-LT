# スライド

## 概要

VRChat 物理学集会の LT「ゲーミング GPU で倍精度の行列積を速くする 〜整数の Tensor Core で FP64 を計算する Ozaki scheme〜」のスライド。
本文は `index.html` (Markdown を `<textarea>` に書く)。数値の出どころはリポジトリ直下の `results/`、`results-phi/`、`results-fortran/` で、各スライドの発表者ノート (`Note:` 以下) に、どの実行の値かを書いてある。

[lt-slide-template](https://github.com/s-sasaki-earthsea-wizard/lt-slide-template) の「水彩の地図帳」テーマ (2026-11、commit `45cd3ad`) を使っている。

## 開発環境

[reveal.js](https://github.com/hakimel/reveal.js) 5.1.0 によってスライドの作成をしています。
Node.js, npmが利用できる環境で、以下のコマンドで開発環境の構築ができます:

```bash
npm install
```

## 使い方

### スライド作成

リポジトリ直下で次を流すと、`slides-jp/` を Python の http.server で配信する (npm は要らない):

```bash
make slides                        # http://localhost:8000
make slides SLIDES_PORT=8080       # ポートを変える
make slides SLIDES_BIND=0.0.0.0    # 別のマシンのブラウザから見る (LAN に公開される)
```

ブラウザで [http://localhost:8000](http://localhost:8000) にアクセスすると、スライドをプレビューできます。
既定では 127.0.0.1 にだけ bind するので、リモートの開発機で立てたときは VS Code のポート転送か SSH トンネルで見る。
`S` キーで発表者ノートが開きます。

ファイルを保存したら自動で再読み込みしたいときは、`slides-jp/` で `npm install` のあと `npm start` (gulp のライブリロード)。

### デザイン

1280×720 (16:9) の「水彩の地図帳」テーマ。各スライドの下端に、章名・進捗バー・ページ番号のフッターが出る。
章の区切り方や部品の使い方は、リポジトリ直下の `CLAUDE.md` の「スライド作成」にまとめてある。

- 表紙のイラストは `assets/images/title.png` を 16:9 の絵に差し替える (今は白一色のプレースホルダー。下 192 px は題の帯で隠れる)
- 配色を変えるときは `styles/custom-style.css` の `--atlas-*` 変数を変える
- 数式は KaTeX。reveal.js の math プラグインが jsDelivr から KaTeX を読み込むので、数式の表示にはネットワークが要る

### フォントのサブセットを作り直す

見出しと本文のフォントは、スライドで使う字だけに絞って `vendor/fonts/` に同梱している (オフラインで表示するため)。
スライドを書いたら、リポジトリ直下で次を流す。google/fonts から元の TTF を取ってきて (`.cache/fonts/`、git の対象外)、
`index.html` で使っている字とかな・英数・記号に絞る。[uv](https://docs.astral.sh/uv/) が要る。

```bash
make fonts
```

### スライドをpdfでエクスポート

`decktape`を使ってスライドをPDFにエクスポートできます。

1. `decktape`のインストール

```bash
npm install -g decktape
```

1. PDFエクスポート

以下のコマンドで`html`で書いたスライドをpdfにエクスポートできます:

``` bash
# スライドは 16:9
decktape --size 1920x1080 index.html slides.pdf
```

## 免責事項

- 本スライドの内容に従ったいかなる結果において著者は一切の責任を負いません

## ライセンスとクレジット

| 対象 | 条件 |
|---|---|
| reveal.js (`dist/`、`plugin/`、`css/`、`js/` など) | MIT License, Copyright (C) 2011-2024 Hakim El Hattab |
| フォント (`vendor/fonts/`): Shippori Mincho B1、Zen Maru Gothic | SIL Open Font License 1.1。本文は同じフォルダの `OFL-*.txt` |
| テーマ (`styles/custom-style.css`) と `index.html` のフッター処理 | lt-slide-template (Apache License 2.0 またはそれ以降) |
| 画像 (`assets/images/`) | 使っているのは `icon_circle.png` (発表者のアバター) と `qrcode.png` (物理学集会)。`title.png` は白一色のプレースホルダー (lt-slide-template のもの) |
