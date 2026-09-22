from pathlib import Path
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'results/v11_claim_audit_0923'
VAULT = Path('/home/issan/Projects/obsidian-research')
intro = '''# V11 主張と証拠の監査（0923）

親: [[V11通し稿_0922]] / 入口: [[V11論文_目次_0921]] / 構成: [[V11構成ボード_0923]] / 残り: [[V11残作業リスト_0922]]

**状態: 主張と根拠の監査を完了。本文への訂正反映はまだ行っていない。** Issa の指示は、骨組みは本人が決め、先に主張が証拠を超えていないか確認し、Vault の根拠をリンクで示すこと。続けて、通し稿を意味のまとまりごとに一文へ圧縮し、組み替えられる構成ボードを作ること。構成・保留文の採用をこの監査で決めない。文献照合のやり直しや新規訓練は行っていない。

## 監査の結論と使い方

中心化・両層の成長上限・凍結窓・応答場交換・5+1の主要な対応比較は、条件を付けて再利用できる。ただし「全部がVaultに合う」という結果ではない。**原稿だけの誤記、Vaultから引き継がれた誤り、観測を一般的な機構へ強めた記述**を分けて記録した。Vault内の説明が食い違う場合は、仕様・保存結果・実装を優先し、その食い違い自体を明示する。

優先して直す具体例:

| 原稿の記述 | 照合結果 | 詳細 |
|---|---|---|
| 応答移植のrは「出力headの初期化」 | 実際はAdamの全m/v/tcの初期化。92.8%はその副腕 | R1 |
| LNは「全ゲートが正側になり線形化して死ぬ」 | 診断列はLN前のz。実際のゲートの証拠にならず、保存zerooutとも整合しない | G1 |
| Cのr=.121・全画像で片側 | Cのrは丸め誤差の範囲で0。登録した片側は99%超で同符号 | G2 |
| Hは第1層の出力だけ、CHBはb2減衰だけ | Hは両隠れ層の出力。Bはb1除去＋b2減衰 | G3 |
| 5+1でHがKKT1/L2 Initに−.106/−.111 | この比較はCH。H−Rと混ざっている | A6 |
| leaky .1のstd低下1.8pt | 保存summaryは1.4pt。幅EMAも実際は分散EMAの平方根 | A8 |
| neffdirの予測の当否 | 付録HでIssaとClaudeの補償/F.elu予測が逆転 | R6 |

これらは主に訂正・限定の問題であり、追加の学習実験を監査の完了条件にはしない。**未同定の機構を論文の中心命題として残したい場合**は別で、例えば「片側化が劣化を媒介する」「微分の床だけが必要」「n_eff単独で能力を決める」「LNは正側で線形化して死ぬ」を主張するには、現在の証拠では足りない。LNの診断は保存状態の再解析が候補であり、ただちに新規訓練を要求する意味ではない。

各項目には原稿の短い引用・位置、根拠のVaultノートと見出し、言える範囲、最小修正文を示す。修正文は**未採用の案**。支持された主張も各担当範囲の表に掲載した。構成ボードではこの案を一文に圧縮して表示し、元の文章を併記する。

## 対象と確認範囲

- 対象原稿: [[V11通し稿_0922]]、771行。SHA-256 `8d65672fd89d83f90acf5f7cab8aa472f693659dff9d1f9105e84580cc55b40a`。以下の行番号はこの版に固定。
- Vault head: `8ac9fefe37e8df97666a9ce4b30284a9bbb2a330`。先生側の未確認変更は無し。
- 実験repoの参照head: `3c37418`。主張と原記録が合わない箇所で既存仕様・保存CSV・コードへ遡った。
- 要旨・序論・§2–§8・方法・限界・図表説明・付録の自研究の主張を確認。文献そのものの書誌・原典照合は前回の完了範囲を引き継ぎ、再実施していない。
- 著者側の複数担当による照合であり、独立第三者査読・全実験の再実行ではない。
- 現在の通し稿と50ページPDFは監査前の版。今回の指摘を反映した完成稿とは扱わない。

---

'''
parts=[]
for key,label in [('root','横断・導出・手続き'),('geometry_drive','配置・中心化・駆動・成長'),('activation','活性化・実ラベル比較'),('response','応答と再学習・予測記録')]:
    s=(R/f'{key}.md').read_text()
    s=s.split('\n',1)[1].lstrip()
    parts.append(f'## 監査領域：{label}\n\n{s}')
result=intro+'\n\n---\n\n'.join(parts)+'\n'
(R/'audit.md').write_text(result)

# Verify exact note paths and heading anchors, and retain hashes of cited sources.
issues=[]; refs=[]
for target in re.findall(r'\[\[([^\]]+)\]\]',result):
    target=target.split('|')[0]
    note,sep,heading=target.partition('#')
    if '/' in note:
        paths=[VAULT/(note if note.endswith('.md') else note+'.md')]
        paths=[p for p in paths if p.exists()]
    else: paths=list(VAULT.rglob(note+'.md'))
    if not paths and note in ['V11構成ボード_0923']: continue # to be created in this task
    if len(paths)!=1:
        issues.append({'target':target,'error':'missing_or_ambiguous','paths':[str(p) for p in paths]}); continue
    p=paths[0];text=p.read_text()
    headings=[re.sub(r'^#{1,6}\s+','',x).strip() for x in text.splitlines() if re.match(r'^#{1,6}\s+',x)]
    if sep and heading not in headings: issues.append({'target':target,'error':'heading_not_exact'})
    refs.append({'path':str(p.relative_to(VAULT)),'heading':heading,'sha256':hashlib.sha256(text.encode()).hexdigest()})
proof={'manuscript_sha256':hashlib.sha256((R/'manuscript_snapshot.md').read_bytes()).hexdigest(),'vault_head':'8ac9fefe37e8df97666a9ce4b30284a9bbb2a330','repo_head':'3c37418','links_checked':len(refs),'issues':issues,'sources':refs,'manuscript_unchanged':(VAULT/'可塑性喪失/論文作成/V11通し稿_0922.md').read_bytes()==(R/'manuscript_snapshot.md').read_bytes()}
(R/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='sources'},ensure_ascii=False))
