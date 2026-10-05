#!/usr/bin/env bash
#
# 仓库内 shell 脚本的静态检查。
#
# 只有两条规则，但两条都对应**真实踩过的坑**：
#
#  ① 语法必须过（`bash -n`）。
#
#  ② `$VAR` 后面**不能紧跟非 ASCII 字符**。在 UTF-8 环境下 bash 会把那些字节
#     也算进变量名 —— `"$t：跳过"` 会被解析成引用变量 `t：`，运行时报
#     `t：: unbound variable`，而且只在**跑到那一行**时才炸（前面看着一切正常）。
#     这个仓库的注释与提示语全是中文，所以这条不是假想风险：
#     2026-10-05 一天之内在 check-arches.sh 与 verify-artifact.sh 各踩了一次。
#     正确写法是 `${t}：跳过`。
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0
for f in scripts/*.sh; do
  if ! bash -n "$f" 2>/dev/null; then
    echo "!! 语法错误：$f"
    bash -n "$f" || true
    fail=1
  fi
done

# ② 只查**代码行**（整行以 # 开头的注释不参与执行，但仍会被这里的正则命中，
#    所以先剔掉纯注释行，避免把"讲这条规则的那行注释"自己报出来）。
python3 - <<'PY' || fail=1
import glob
import re
import sys

# $VAR 紧跟一个非 ASCII 字符
pat = re.compile(r'\$[A-Za-z_][A-Za-z0-9_]*[^\x00-\x7f]')
bad = []
for path in sorted(glob.glob('scripts/*.sh')):
    for i, line in enumerate(open(path, encoding='utf-8'), 1):
        stripped = line.lstrip()
        if stripped.startswith('#'):
            continue
        for m in pat.finditer(line):
            bad.append(f'{path}:{i}: {m.group(0)!r}\n    {line.strip()}')

if bad:
    print('!! $VAR 后面紧跟了非 ASCII 字符（bash 会把它算进变量名）—— 改成 ${VAR}：')
    for b in bad:
        print('   ', b)
    sys.exit(1)
print('✅ shell 脚本检查通过')
PY

exit "$fail"
