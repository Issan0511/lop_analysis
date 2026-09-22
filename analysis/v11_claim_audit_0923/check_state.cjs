// Exercise the actual board's persistence validators and Markdown serializer.
// This does not control a browser or access browser storage.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'results/v11_claim_audit_0923/board_data.json'), 'utf8'));
const html = fs.readFileSync(path.join(root, 'output/structure/V11_structure_board_0923.html'), 'utf8');
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Full application syntax, including UI event handlers.
const prefix = script.slice(0, script.indexOf("  $('undo').addEventListener"));
const sandbox = {document:{getElementById:id=>{
  assert.equal(id,'board-data'); return {textContent:JSON.stringify(data)};
}}};
vm.createContext(sandbox);
vm.runInContext(prefix+`state=makeInitial(); globalThis.check={makeInitial,validateImport,envelope,markdown,set:s=>{state=s}};})();`,sandbox);
const api=sandbox.check;
const plain=x=>JSON.parse(JSON.stringify(x));
const original=plain(api.envelope());
assert.deepEqual(plain(api.validateImport(original)),original.state);
const modified=structuredClone(original);
modified.state.centralMessage='構成のテスト <script>alert(1)</script>';
modified.state.order.reverse();
modified.state.cards.abstract={placement:'appendix',claim:'検証用の一文。',memo:'メモ\n二行目'};
api.set(api.validateImport(modified));
assert.deepEqual(plain(api.envelope()),modified);
assert.match(api.markdown(),/検証用の一文。/);
assert.match(api.markdown(),/編集済みの文/);
assert.match(api.markdown(),/メモ\n二行目/);
assert.match(api.markdown(),/V11主張と証拠の監査_0923/);
let rejected=0;
for(const mutate of [
 x=>{x.version='wrong'},x=>{x.manuscriptSha='wrong'},
 x=>{x.state.order[0]=x.state.order[1]},x=>{x.state.order.pop()},
 x=>{x.state.cards.abstract.placement=['body']},
 x=>{x.state.cards.abstract.placement='unknown'},
 x=>{x.state.cards.abstract.claim=42},
 x=>{x.state.decisions.b2b8.choice='unregistered'},
 x=>{x.state.preset='unknown'},
]){
 const invalid=structuredClone(modified);mutate(invalid);
 assert.throws(()=>api.validateImport(invalid));rejected++;
 assert.deepEqual(plain(api.envelope()),modified,'Rejected import must not alter current state');
}
console.log(JSON.stringify({syntax:'pass',roundtrip:'pass',markdown:'pass',invalid_imports_rejected:rejected,cards:data.cards.length}));
