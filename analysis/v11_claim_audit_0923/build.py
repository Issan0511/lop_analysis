"""Build an offline author-owned structure board from the audited claim inventory."""
from pathlib import Path
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/v11_claim_audit_0923'
OUT = ROOT / 'output/structure/V11_structure_board_0923.html'
VAULT = Path('/home/issan/Projects/obsidian-research')
MANUSCRIPT = '可塑性喪失/論文作成/V11通し稿_0922.md'
AUDIT = '可塑性喪失/論文作成/V11主張と証拠の監査_0923'
source = (RESULT / 'manuscript_snapshot.md').read_text()
lines = source.splitlines()
reports = {k:(RESULT / f'{k}.md').read_text() for k in ['root','geometry_drive','activation','response']}
flags = {}
for key, report in reports.items():
    matches = list(re.finditer(r'^#{2,3} ([TGAR]\d+)\.?\s*(.*)$', report, re.M))
    for i,m in enumerate(matches):
        content = report[m.end():matches[i+1].start() if i+1<len(matches) else len(report)]
        # Stop at the next major supporting/concluding heading.
        content = re.split(r'^## (?:支持|維持|そのまま|構成|小さな|引用|結論)',content,flags=re.M)[0].strip()
        flags[m[1]] = {'id':m[1], 'title':m[2].strip(), 'detail':content, 'suggestion':'詳細中の最小修正文は提案です。原稿への反映・採用は未実施。'}

def evidence(path, heading='', label=''):
    return {'path':path.removesuffix('.md'), 'heading':heading, 'label':label or path.split('/')[-1].removesuffix('.md')}

def note(name, heading='', label=''):
    paths=list(VAULT.rglob(name+'.md'))
    if len(paths)!=1: raise ValueError((name,paths))
    return evidence(str(paths[0].relative_to(VAULT)),heading,label)

def card(id,title,claim,scope,grade,figures,refs,ids,original_lines,placement='body'):
    return dict(id=id,title=title,claim=claim,scope=scope,grade=grade,figures=figures,
        sources=[note(*r) for r in refs],flags=[flags[x] for x in ids],
        original='\n\n'.join(f'L{n}: {lines[n-1]}' for n in original_lines),defaultPlacement=placement)

cards=[
card('geometry','配置を記述する幾何',
 '前活性は入力平均への投影で決まる共通位置と、入力ごとの揺らぎに分かれる。重み・上流入力・biasの変化を分けて記述できる。',
 '恒等式。負の向きや、因果の唯一性は式だけからは出ない。', '恒等式', ['図1a'],
 [('中心主張v11草案_0920','0.5 幾何（恒等式）'),('駆動源問題_0909','12.5 Adamでは履歴と実際の分母を含めて保証する')],['T1','T2'],[25,68]),
card('initial','初期の配置を予測する',
 'raw・std・入力中心化Cの第1層の初期片側率は、登録した近似の予測帯と両立した。γの中間条件と第2層では外れた。',
 '未使用20 seed・訓練なし。片側は99%超の画像で同符号。Cのrは約0。全8条件の主判定はL1_MODEL_MISS。','登録・近似モデル',['図1b','付録C'],
 [('CIFAR初期配置のr予測_A5_結果_0920',)],['G2'],[70,72]),
card('centering','中心化で経路を塞ぐ',
 '乱数ラベルCIFARのraw ReLUでは、入力中心化Cに両隠れ層出力の中心化Hを加えると、後期の再学習成績が0.113から0.987へ上がった。',
 'raw・ReLU・50課題・後期t31–50。Hは両隠れ層出力のEMA中心化。初回から学べないrefは進行的LoPの主対照にしない。','登録された介入',['図2','表2','付録E'],
 [('ReLUが沈む道を1本ずつ塞ぐ_relu_doors_結果_0919','1. 窓（t31–50 の online、seed 中央値・10 seed）'),('H単独はReLUを救うか_relu_doors_h_ref_S2_結果_0920','3. 登録結果')],['G1','G3'],[92,94]),
card('bias','長い時間で残るbiasの経路',
 '200課題ではCHの登録COLLAPSEが7/10だった。第1層biasの除去と第2層bias減衰を加えたCHBは10/10で性能を保った。',
 '7/10は性能・深さの複合判定（両方6・深さのみ1）。総合INCONCLUSIVE。予防履歴を含む二つの操作で、b2だけの独立効果ではない。','登録・総合INCONCLUSIVE',['図3','表2'],
 [('H単独はReLUを救うか_relu_doors_h_ref_S2_結果_0920','7. V11 での位置'),('V11証拠台帳_§1-§5_0921','§1 第一の柱 — μ の大きさと配置')],['G3','G4'],[150]),
card('drive','切替直後の自己更新',
 'C腕ReLUの課題2–5では、自己更新による負の正味変位が切替直後75更新に集中し、後続の上流変位は全5 seedで負だった。',
 'CIFAR・C腕・ReLU・第2層・t2–5。登録した十分条件の頻度と、副記述の変位収支を区別する。','登録＋収支の記述',['図5','付録D'],
 [('CIFAR課題間のAdam駆動_A2_結果_0920',),('駆動源問題_0909','12.7 多層の実変位と時間積分に残る条件')],['G6','T1'],[190,192]),
card('freeze','切替直後を凍結する介入',
 'raw CIFARのLEで凍結量を揃えると、課題中央より切替直後の凍結で初期の入力平均成長が小さくなった。後期の救済は部分的だった。',
 'raw・leaky .1→ELU。t5の‖μ₂‖は141対672、10/10。同じ操作のLLは逆向き。完全救済・唯一の源とは言わない。','登録された介入',['図6'],
 [('押しは切替で作られるか_凍結窓_0922','登録判定')],['G6'],[194,196]),
card('growth','重みの成長を制限する',
 'std CIFARのELUでは、両層の行ノルムに上限を置くと後期成績が10.1%から86.0%へ上がった。沈下の主窓の帳簿は自己・上流・交差項の混合だった。',
 'ELU/std・課題1末の半径・t31–50・seed平均。A6は既知軌道の登録再解析、主窓MIXED、後期99%は報告のみ。','登録された介入＋再解析',['図7'],
 [('CIFAR層別成長制限_S5_結果_0920','主窓の結果'),('CIFARの崩壊層と輸送帳簿_A6_結果_0920','主判定')],['G7','T4'],[234,236,238]),
card('class','運ばれた入力の応答が消える条件',
 '指定入力が負側へ運ばれ続けると、負側微分がゼロへ漸近する固定尺度の活性化では局所応答が消える。追加の成長条件の下で、その入力からの生勾配も消える。',
 '輸送は仮定。生勾配には微分の指数減衰と要求・入力の深さに対する多項式上界。Adam実変位や機能的LoPの証明ではない。','条件付き導出',[],
 [('固定尺度のReLU型活性化の飽和不可避性_仮説_0916','3.2 負側への一様輸送を仮定した応答消失'),('固定尺度のReLU型活性化の飽和不可避性_仮説_0916','元の関数クラスとの関係と、結論の射程')],[],[29]),
card('activation','活性化による応答と成績の分岐',
 '乱数ラベルCIFARの50課題では、固定尺度のReLU型4種が床へ落ち、leakyは後期水準が低下した一方、kunekuneと適応Snakeは高い成績を保った。',
 '測った関数・raw/std・有限50課題。片側化は事後観測で、劣化の媒介は未同定。適応尺度だけによる性能保証ではない。','登録比較＋事後診断',['図8','表3','付録F'],
 [('RL-CIFARのMLPで13の活性化をバトル_kunekune_結果_0918','1. 窓（seed 中央値、* = 崩壊）'),('RL-CIFARの入力平均と非線形領域_V10とkunekuneの考察_0919','13.2 三つの主張を分ける')],['A2','A3','A4','A8'],[264,266,268]),
card('chimera','層を替えて適用範囲を確かめる',
 'raw CIFARの比較した7セルでは、両層をleaky .1にした腕だけが生存基準を満たし、片方をELUにすると後期成績が大きく下がった。',
 'LL−LE +.627、LL−EL +.589、各10/10。主判定MIXED。rawのみ、前向き値と微分は未分離。LL/EE/GGは参照の再利用。','登録・MIXED',['付録J'],
 [('層別キメラをRL-CIFARへ_S-A_結果_0922','1. 登録判定'),('層別キメラをRL-CIFARへ_S-A_結果_0922','5. 限定')],['A1'],[745,747,749,750],'appendix'),
card('response','応答の場を変えると再学習が動く',
 'std CIFARのELU第2層で、初期の出力を保つ人工的な応答場交換により、次課題の成績が復元側+46.7pt、沈降側−67.9ptに動いた。',
 '主P1はAdam継続、主P2は両腕Adam reset。rはAdam履歴の初期化。応答場の全媒介・恒久救済・n_eff単独法則は示さない。','登録された両方向介入',['図9'],
 [('CIFAR応答場移植_S4_結果_0920',),('CIFARの崩壊層と輸送帳簿_A6_結果_0920','登録補助窓（REPORT_ONLY）')],['R1','R2','R7'],[312,314,318]),
card('chain','MNISTで応答の回復・維持を分ける',
 'MNISTの応答交換・成長制限・担い手追加の介入は、応答の回復と維持が再学習成績に関わることを支持した。成長の効果は状態によって異なった。',
 'MNIST ELUの同一系列。追加側介入は成立、減少側は補償で操作不成立。0.72等は人工介入差の報告比。n_effは箱をまたがない。','登録介入＋報告・事後解釈',['付録A'],
 [('担い手を直接動かす_n_effの因果の向き_結果_0918','3. 登録判定'),('cap12とrefの網で第2層の場を入れ替える_結果_0917','2. 登録判定'),('動く場の移植_応答低下からLoPへ_結果_0917','9. 言えないこと')],['R3','R4','R5','R6','G7'],[316,524,526,529],'appendix'),
card('real','実ラベルでも比較する',
 '5+1 CIFARの後期中央値ではSnake系が上位に入り、20 seedでKKT1とSNAは登録帯±0.5ptの同等性を満たした。SNAは原典λのL2 Initより優れ、調整λとは同等だった。',
 'MLP・780更新。SNA対13腕の優位とKKT1–SNAの同等性を区別。全腕で早期後期の低下と正のfresh gap。増予算でもRのgapは残る。','登録比較・同等性',['図4','図10','表4','付録B'],
 [('5+1CIFARをMLPで_16腕とl2initと扉と予算_結果_0920','3. 追補 2・3: L2 Init（`R` + λ‖θ − θ₀‖²）'),('5+1CIFARをMLPで_16腕とl2initと扉と予算_結果_0920','4. 追補 4: relu_doors の扉をこの箱へ'),('5+1CIFARをMLPで_16腕とl2initと扉と予算_結果_0920','5. 追補 5: 予算 ×10（遅さか到達点か）')],['A4','A5','A6','A7','G5'],[336,340]),
card('limits','主張する範囲を明示する',
 '負側飽和を介する経路を、主に正規化層なし二隠れ層MLP・Adamで調べた。各介入が支持する条件を個別に示す。',
 '三設定すべてで同じ因果の鎖を閉じたわけではない。一般の不可逆性・無限時間保持・CNN/SGDへの一般化はこの証拠には含まれない。','対象範囲・限定',['表5'],
 [('V11通し稿_0922','11 限界'),('中心主張v11草案_0920','6. 証拠表（柱 × 箱）')],['T5','T6','T4'],[450,452,458,460,462]),
]

# Keep the source manuscript order. The thematic inventory only supplies evidence;
# it is not a proposed replacement outline.
themes={c['id']:c for c in cards}
blocks=json.loads((RESULT/'compressed_blocks.json').read_text())
refs_by_block={
 'conditional_derivation':['class'], 'drive_initial':['drive'],
 'abstract':['geometry','centering','growth','response','real'],
 'intro_geometry':['geometry','centering'], 'pending_b2_b8':['geometry'],
 'intro_mechanism':['class','activation','response'], 'setting':['limits'],
 'geometry_identity':['geometry'], 'initial_geometry':['initial'],
 'centering_doors':['centering'], 'bias_long_run':['bias'], 'real_doors':['real'],
 'drive_windows':['drive'], 'pending_decision9':['drive'], 'freeze_windows':['freeze'],
 'growth_ledger':['growth'], 'growth_caps':['growth','chain'],
 'fixed_activation':['activation','class'], 'layer_chimera':['chimera'],
 'leaky_activation':['activation'], 'adaptive_activation':['activation'],
 'response_definition':['response','class'], 'response_swap':['response'], 'mnist_chain':['chain'],
 'real_activation':['real'], 'real_l2init':['real'], 'real_budget':['real'],
 'related_centering':['centering'], 'related_activation':['activation','real'],
 'methods_activation':['activation'], 'methods_centering':['centering','bias'],
 'methods_interventions':['chimera','freeze','growth'], 'methods_response_controls':['response','real'],
 'methods_measurement':['response','growth','activation'],
 'methods_procedure':['limits','real','initial'],
 'limits_general':['limits','centering','real'],
 'limits_mechanism':['chimera','freeze','initial','drive'],
 'limits_implementation':['response','limits']}
flags_by_block={
 'conditional_derivation':['T3'], 'drive_initial':['G6'],
 'abstract':['T5'], 'intro_geometry':['T1','T2'], 'pending_b2_b8':['T1','T2'],
 'intro_mechanism':['T3','A1','R2','T5'], 'setting':['R2'],
 'geometry_identity':['T1'], 'initial_geometry':['G2'],
 'centering_doors':['G1','G3'], 'bias_long_run':['G3','G4'], 'real_doors':['G5','A6'],
 'drive_windows':['G6'], 'pending_decision9':['T1'], 'freeze_windows':['G6'],
 'growth_ledger':['G7','T4'], 'growth_caps':['G7'],
 'fixed_activation':['R7'], 'layer_chimera':['A1'],
 'leaky_activation':['A2','A8'], 'adaptive_activation':['A2','A3','A4'],
 'response_definition':['R2'], 'response_swap':['R1','R2'], 'mnist_chain':['R3','R4','R5','R6','R7'],
 'real_activation':['A4','A5'], 'real_l2init':['A4'], 'real_budget':['A6','A7'],
 'related_centering':['G4'], 'related_activation':['A4'],
 'methods_activation':['A8'], 'methods_centering':['G3'], 'methods_interventions':[],
 'methods_response_controls':['R1'], 'methods_measurement':['G1','A8','R7'],
 'methods_procedure':['T4','A4'], 'limits_general':['T5','T6','G4','A7'],
 'limits_mechanism':['A1','G2','G6'], 'limits_implementation':['R2']}
cards=[]
for block in blocks:
    refs=[evidence(MANUSCRIPT, block['section'] if block['section']=='要旨' else '',
                  '元の通し稿（'+str(block['sourceStart'])+'–'+str(block['sourceEnd'])+'行）')]
    for theme in refs_by_block[block['id']]:
        for ref in themes[theme]['sources']:
            if ref not in refs: refs.append(ref)
    if block['id']=='drive_initial':
        refs.append(note('沈降と回復の競争を検証_erosion_race_結果_0919','3. 登録結果'))
    if block['theme']=='related':
        # Literature checking was already done by the author; this audit did not repeat it.
        refs[0]['label']='元の通し稿・関連研究（文献再照合は今回の対象外）'
    cards.append(dict(block,sources=refs,
        flags=[flags[k] for k in flags_by_block[block['id']]],
        original='\n'.join(f'L{i+1}: {lines[i]}' for i in range(block['sourceStart']-1,block['sourceEnd']))))
order=[c['id'] for c in cards]
def themed_order(theme_order):
    ranks={theme:i for i,theme in enumerate(theme_order)}
    return [c['id'] for c in sorted(cards,key=lambda c:ranks.get(c['theme'],len(ranks))) ]
data={'version':'2026-09-23','manuscriptSha':hashlib.sha256(source.encode()).hexdigest(),
 'auditNote':{'title':'V11 主張と証拠の監査（0923）','path':AUDIT},'cards':cards,
 'presets':[
  {'id':'mechanism','title':'機構の順','description':'配置 → 駆動 → 成長 → 活性化 → 再学習の順に説明する。','tradeoff':'異なる箱・条件をつなぐ箇所で、各根拠の範囲を明示する必要がある。','order':themed_order(['abstract','intro','setting','geometry','initial','centering','bias','drive','freeze','growth','class','activation','chimera','response','chain','real','related','methods','limits'])},
  {'id':'intervention','title':'介入の順','description':'中心化・凍結・上限・応答交換で、何が変わったかを先に見せる。','tradeoff':'幾何や条件付き導出を後で説明するため、用語の短い導入が必要。','order':themed_order(['abstract','intro','setting','centering','bias','freeze','growth','response','real','geometry','initial','drive','class','activation','chimera','chain','related','methods','limits'])},
  {'id':'activation-first','title':'活性化の分岐から','description':'応答と成績の違いを問いにして、配置・進行・応答介入へ進む。','tradeoff':'性能比較と機構の同定を分け、適応尺度だけを原因にしない説明が必要。','order':themed_order(['abstract','intro','setting','activation','class','real','chimera','geometry','initial','centering','bias','drive','freeze','growth','response','chain','related','methods','limits'])}
 ],
 'decisions':[
  {'id':'b2b8','title':'B2・B8：駆動と戻る経路の文','status':'保留。Adamの履歴と一標本SGDの条件は監査T1/T2参照。','options':['条件を明記した文へ直す','本文から外して補足へ','判断を保留する']},
  {'id':'d9a','title':'決定9a：駆動源の理論','status':'先生の稿との関係は未決。','options':['本稿は測定と限定を自前で書く','先生の稿を参照して要約する','範囲外を維持して後で参照を足す']},
  {'id':'d9b','title':'決定9b：W増大の理論','status':'Issaの理論を本稿に入れるかは未決。','options':['本稿では範囲外にする','成立した範囲を本稿に入れる','別稿として進める']}
 ]}
RESULT.mkdir(parents=True,exist_ok=True)
(RESULT/'board_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
template=ROOT/'analysis/v11_claim_audit_0923/board.template.html'
if template.exists():
    text=template.read_text()
    assert text.count('__BOARD_DATA__')==1
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(text.replace('__BOARD_DATA__',json.dumps(data,ensure_ascii=False).replace('<', r'\u003c')))
    print(OUT)
else: print('Data ready; board template pending.')
print(f'{len(cards)} cards, {len(flags)} audit items; SHA {data["manuscriptSha"]}')
