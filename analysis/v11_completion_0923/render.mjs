// Render the completed Markdown snapshot; no experiment is run.
// npm install --prefix <deps> katex@0.16.22 markdown-it@14.1.0
// node render.mjs <deps>/node_modules <playwright-module-root>
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const req=createRequire(import.meta.url);
const deps=process.argv[2];
const MarkdownIt=req(path.join(deps,'markdown-it'));
const katex=req(path.join(deps,'katex'));
const {chromium}=req(path.join(process.argv[3],'playwright'));
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const out=path.join(root,'output/pdf');fs.mkdirSync(out,{recursive:true});
let source=fs.readFileSync(path.join(root,'results/v11_completion_0923/V11_integrated_0923.md'),'utf8');
source=source.split('## 編集・照合記録')[0];
source=source.replace(/^親:.*\n/m,'').replace(/^読み方:.*\n/m,'');
source=source.replace(/\[\[([^\]]+)\]\]/g,'$1');
source=source.replace(/\[(\d+)\](?!\()/g,(a,n)=>`<a class="cite" href="#ref-${n}">[${n}]</a>`);
source=source.replace(/^<a class="cite" href="#ref-(\d+)">/gm,'<a id="ref-$1" class="cite" href="#ref-$1">');
const math=[];
source=source.replace(/\$\$([\s\S]*?)\$\$|\$([^\n$]+)\$/g,(a,display,inline)=>{
 const i=math.length; math.push(katex.renderToString(display??inline,{displayMode:display!==undefined,throwOnError:true,output:'html',strict:'ignore'}));
 return `MATHPLACEHOLDER${i}END`;
});
const md=new MarkdownIt({html:true,breaks:false,typographer:false});
source=source.replace(/<!-- figure:([^ ]+) -->\n([\s\S]*?)<!-- \/figure -->/g,(a,n,body)=>{
 const png=body.match(/\]\(([^)]+\.png)\)/)[1];
 const file=path.join(root,'results/v11_figures_0921',path.basename(png));
 const bytes=fs.readFileSync(file);const width=bytes.readUInt32BE(16),height=bytes.readUInt32BE(20);
 const orientation=width/height>=2.0?'wide':'portrait';
 const caption=body.split('\n').filter(x=>x.startsWith('> ')).map(x=>x.slice(2)).join(' ');
 return `<figure class="${orientation}" data-figure="${n}"><img alt="図 ${n}" src="data:image/png;base64,${bytes.toString('base64')}"/><figcaption>${md.renderInline(md.utils.escapeHtml(caption))}</figcaption></figure>\n`;
});
let content=md.render(source);
content=content.replace(/MATHPLACEHOLDER(\d+)END/g,(a,n)=>math[Number(n)]);
content=content.replace(/(<h3>表 [1-5][\s\S]*?)(?=<h[23]>|<figure|$)/g,'<section class="table-section wide">$1</section>');
content=content.replace(/(<h2>付録 [GH][\s\S]*?)(?=<h2>|$)/g,'<section class="table-section wide">$1</section>');
content=content.replace(/(<h2>付録 [A-FJ][^<]*<\/h2>)\s*(<figure[^>]*>)/g,'$2$1');
content=content.replace(/【(B2・B8|決定 9)([\s\S]*?)】/g,'<span class="pending">【$1$2】</span>');
let css=fs.readFileSync(path.join(deps,'katex/dist/katex.min.css'),'utf8');
css=css.replace(/url\(fonts\/([^)]*)\)/g,(a,f)=>`url(data:font/woff2;base64,${fs.readFileSync(path.join(deps,'katex/dist/fonts',f)).toString('base64')})`);
const html=`<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><title>V11 図表統合稿 0923</title><style>${css}
@page {size:A4 portrait;margin:17mm 17mm 19mm;}
@page wide {size:A4 landscape;margin:15mm 17mm 19mm;}
@page portrait {size:A4 portrait;margin:17mm 17mm 19mm;}
*{box-sizing:border-box} body{font:10.5pt/1.75 'Noto Sans CJK JP',sans-serif;color:#18222d;margin:0;}
h1{font-size:22pt;line-height:1.4;margin:8mm 0 6mm;color:#163b50;}
h2{font-size:15pt;line-height:1.5;border-bottom:1px solid #b4c5cc;padding-bottom:2mm;margin:8mm 0 4mm;break-after:avoid;}
h3{font-size:12pt;margin:6mm 0 3mm;break-after:avoid;}
p{margin:0 0 3.4mm;orphans:3;widows:3;} a{color:#285d76;text-decoration:none;} strong{font-weight:600;}
code{font:8.3pt/1.4 'Noto Sans Mono CJK JP',monospace;overflow-wrap:anywhere;}
.pending{background:#fff1ce;border-bottom:1px solid #c19a41;}
.wide{page:wide}.portrait{page:portrait}
figure{margin:0;break-before:page;break-after:page;break-inside:avoid;display:block;}
figure img{display:block;width:100%;object-fit:contain;object-position:left top;}
figure.wide img{max-height:116mm;} figure.portrait img{max-height:174mm;}
figure h2{margin-top:0;}
figcaption{font-size:9pt;line-height:1.6;margin-top:4mm;overflow-wrap:anywhere;}
.table-section{break-before:page;break-after:page;}.table-section h2,.table-section h3{margin-top:0;}.table-section p{font-size:9pt;}
table{border-collapse:collapse;width:100%;font-size:9pt;line-height:1.55;table-layout:auto;}
thead{display:table-header-group;}tr{break-inside:avoid;}th,td{padding:2.3mm 2mm;text-align:left;vertical-align:top;border-bottom:1px solid #d1dde2;overflow-wrap:anywhere;}
th{background:#e7eef2;color:#163b50;font-weight:600;} tbody tr:nth-child(even){background:#f7f9fa;}
table code{font-size:8pt;} table p{margin:0;}.katex{font-size:1.02em;}.katex-display{margin:5mm 0;}
blockquote{margin:3mm 0;padding-left:4mm;border-left:2px solid #c8d5da;font-size:9.5pt;} hr{border:0;border-top:1px solid #d1dde2;margin:5mm 0;}
li{margin-bottom:2mm;}.notice{font-size:9pt;color:#556773;margin-bottom:6mm;}
</style></head><body><div class="notice">2026-09-23 / 日本語・図表統合稿 / 黄色の【 】は判断保留</div>${content}</body></html>`;
const htmlPath=path.join(out,'V11_review_0923.html');fs.writeFileSync(htmlPath,html);
const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
const page=await browser.newPage();await page.goto('file://'+htmlPath,{waitUntil:'load'});await page.evaluate(()=>document.fonts.ready);
const errors=await page.evaluate(()=>({brokenImages:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.alt),mathErrors:document.querySelectorAll('.katex-error').length,figures:document.querySelectorAll('figure').length,tables:document.querySelectorAll('table').length,refs:document.querySelectorAll('[id^="ref-"]').length}));
if(errors.brokenImages.length||errors.mathErrors||errors.figures!==17||errors.refs!==13)throw Error(JSON.stringify(errors));
await page.pdf({path:path.join(out,'V11_review_0923.pdf'),printBackground:true,preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<div></div>',footerTemplate:'<div style="font-size:8px;width:100%;text-align:center;color:#667788;">V11 · 2026-09-23 · <span class="pageNumber"></span> / <span class="totalPages"></span></div>'});
fs.writeFileSync(path.join(root,'results/v11_completion_0923/render_checks.json'),JSON.stringify(errors,null,2)+'\n');
await browser.close();console.log(JSON.stringify(errors));
