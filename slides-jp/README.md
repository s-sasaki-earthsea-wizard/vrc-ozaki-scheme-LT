# スライド

## 概要

VRChat 物理学集会の LT「物理シミュレーションの強い味方!? GeForce GPUで尾崎スキームを試してみた! 〜速い！正確！でも……？〜」のスライド。
本文は `index.html` (Markdown を `<textarea>` に書く)。数値の出どころはリポジトリ直下の `results/`、`results-phi/`、`results-fortran/`、`results-cusolver/`、`results-adi/` で、各スライドの発表者ノート (`Note:` 以下) に、どの実行の値かを書いてある。

[lt-slide-template](https://github.com/s-sasaki-earthsea-wizard/lt-slide-template) の「水彩の地図帳」テーマ (2026-11、commit `45cd3ad`) を使っている。

## 開発環境

[reveal.js](https://github.com/hakimel/reveal.js) 5.1.0 によってスライドの作成をしています。
Node.js, npmが利用できる環境で、以下のコマンドで開発環境の構築ができます:

```bash
npm install
```

## 使い方

### スライド作成

リポジトリ直下で次を流すと、`slides-jp/` を Python の http.server で配信する (npm は要らない)。
配信は `scripts/serve-slides.py` で、`Cache-Control: no-store` を返すので、差し替えた画像やフォントも普通の再読み込みで反映される:

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

- 表紙のイラストは `assets/images/title.png` (16:9)。題の帯より上 (1280×528) に敷くので 192 px 切れる。どこを見せるかは `index.html` の img の `object-position` で選ぶ (今は上を 90 px 切っている)。題を `<br>` で 2 行にすると、帯に収まるよう文字が小さくなる
- 配色を変えるときは `styles/custom-style.css` の `--atlas-*` 変数を変える
- 数式は KaTeX。reveal.js の math プラグインが jsDelivr から KaTeX 0.19.1 を読み込むので、数式の表示にはネットワークが要る。
  版は `index.html` の `Reveal.initialize` (`katex: { version: ... }`) で固定している

### フォントのサブセットを作り直す

見出しと本文のフォントは、スライドで使う字だけに絞って `vendor/fonts/` に同梱している (オフラインで表示するため)。
スライドを書いたら、リポジトリ直下で次を流す。google/fonts から元の TTF を取ってきて (`.cache/fonts/`、git の対象外)、
`index.html` で使っている字とかな・英数・記号に絞る。[uv](https://docs.astral.sh/uv/) が要る。

```bash
make fonts
```

### スライドを PDF でエクスポート

[decktape](https://github.com/astefanutti/decktape) でスライドを PDF に書き出す。

最初に 1 回、`decktape` を入れる (3.17.0 は Node.js 22.12 以降が要る)。Puppeteer 用の Chrome も一緒にダウンロードされる:

```bash
npm install -g decktape
```

リポジトリ直下で次を流すと、`slides-jp/slides-jp.pdf` ができる (git の対象外)。`slides-jp/` で `make` を流しても同じ:

```bash
make pdf
```

- 数式の KaTeX を jsDelivr から読むので、書き出しにはネットワークが要る
- スライドの文字を変えたときは、先に `make fonts` を流す
- Ubuntu 23.10 以降では、AppArmor の制限で Puppeteer の Chrome がサンドボックスを起動できず、`No usable sandbox!` で落ちる。
  そのため Linux では `--no-sandbox` を付けている。信頼できるスライドだけを書き出す
- `DECKTAPE_ARGS` を渡すと、decktape に渡すオプション (Linux では `--no-sandbox`) が置き換わる。別の Chrome を使うときなどに指定する:

  ```bash
  make pdf DECKTAPE_ARGS=--chrome-path=/usr/bin/google-chrome-stable
  ```

- できた PDF を Ghostscript (10.02.1 で確認) で圧縮すると、太字のマーカーが消える

## 免責事項

- 本スライドの内容に従ったいかなる結果において著者は一切の責任を負いません

## ライセンスとクレジット

| 対象 | 条件 |
|---|---|
| reveal.js (`dist/`、`plugin/`、`css/`、`js/` など) | MIT License, Copyright (C) 2011-2024 Hakim El Hattab |
| フォント (`vendor/fonts/`): Shippori Mincho B1、Zen Maru Gothic | SIL Open Font License 1.1。本文は同じフォルダの `OFL-*.txt` |
| テーマ (`styles/custom-style.css`) と `index.html` のフッター処理 | lt-slide-template (Apache License 2.0 またはそれ以降) |
| 画像 (`assets/images/`) | 使っているのは `title.png` (表紙のイラスト。画像生成 AI で作成)、`icon_circle.png` (発表者のアバター)、`qrcode.png` (物理学集会) |
