const katex = require('katex');
let input = '';
process.stdin.on('data', d => input += d);
process.stdin.on('end', () => {
  const items = JSON.parse(input);
  const out = items.map(it => {
    try {
      return katex.renderToString(it.tex, {
        displayMode: !!it.display, throwOnError: false, strict: false, output: 'html',
      });
    } catch (e) { return '<span style="color:#b00">[math error]</span>'; }
  });
  process.stdout.write(JSON.stringify(out));
});
