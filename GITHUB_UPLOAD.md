# 从服务器将二跳实验代码归档到 GitHub

建议本次使用个人私有仓库保存待整理的技术材料。以下指令在服务器 Bash 中执行；本地 Markdown 需先按同名文件放到服务器仓库根目录。本文不执行远程上传，也不改变现有上游 origin。

## 1. 创建空的私有仓库

在 GitHub 网页新建仓库，例如 `ngfix-twohop-archive`，选择 **Private**，不要勾选自动创建 README、LICENSE 或 .gitignore。记下仓库地址，替换下面的 `YOUR_USERNAME`。

若服务器已安装 GitHub CLI，也可在第 4 步改用 CLI 创建，二选一即可。

## 2. 进入现有服务器仓库并保留版本信息

```bash
cd /data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
set -euo pipefail

git status --short
git remote -v
test -f EXPERIMENT_ARCHIVE.md
test -f GITHUB_UPLOAD.md

# 新增归档分支；若此名字已经存在，换一个后缀。
git switch -c archive/twohop-20261026

mkdir -p archive
git rev-parse HEAD > archive/server_base_commit.txt
git remote -v > archive/remotes_before_archive.txt
git diff --stat HEAD > archive/server_changes_before_archive.txt
```

本地 origin 指向上游 `yuhuifishash/NGFix`；服务器配置以实际输出为准。不要直接把修改推到未核对的 origin。`server_base_commit.txt` 记录提交前基点，最终代码身份由本次新 commit 确定。

如 `git commit` 提示缺少身份，仅在当前仓库设置自己的姓名及 GitHub 邮箱：

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

## 3. 排除数据和构建产物，再提交

向现有忽略文件追加归档规则，不删除原规则：

```bash
cat >> .gitignore <<'EOF'

# Two-hop experiment archive: large artifacts remain on the server.
/data/
/result/
/output/
/build/
__pycache__/
*.pyc
*.bak
*.index
*.fbin
*.ibin
*.u32
*.bin
*.zip
*.tar
*.tar.gz
.env
.env.*
EOF

git add .gitignore EXPERIMENT_ARCHIVE.md GITHUB_UPLOAD.md archive/
git add CMakeLists.txt README.md LICENSE ngfixlib/ rabitqlib/ test/ shell/ analysis/ docs/
```

忽略规则不会自动取消已被 Git 跟踪或之前已暂存的文件；本地核对时 data/result/output 未被跟踪，服务器应以这里的暂存列表为准。

可选：把关键小报告复制到 archive，使 GitHub 中也能看到结果依据。缺失的文件跳过，不复制大型逐事件数据。

```bash
python3 - <<'PY'
from pathlib import Path
import shutil

selected = [
    'result/complete_experiment_report_20261013/完整报告.md',
    'result/m16/report_20261015/report.md',
    'result/ngfix_beam_theory_20261021/NGFix补边利用与纯二跳可行性.md',
    'result/webvid_ab_s17_L200/tables_and_hotspots_20261020/01_results_all_72.csv',
    'result/webvid_ab_s17_L200/tables_and_hotspots_20261020/04_online_usage_72.csv',
    'result/webvid_ab_s17_L200/tables_and_hotspots_20261020/07_label_coverage.csv',
]
for name in selected:
    src = Path(name)
    if not src.is_file():
        print('missing:', name)
        continue
    dst = Path('archive/evidence') / src.relative_to('result')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print('copied:', name)
PY

git add archive/
```

复制的旧报告中的图表相对链接仍指原结果目录；这里只保留文本依据，不表示全部附件已上传。

提交前检查：

```bash
git diff --cached --stat
git diff --cached --name-only

# 检查暂存内容自身，避免已跟踪的大索引绕过忽略规则。
python3 - <<'PY'
import subprocess

raw = subprocess.check_output(['git', 'diff', '--cached', '--name-only',
                               '--diff-filter=ACMRT', '-z'])
large = []
for item in raw.split(b'\0'):
    if not item:
        continue
    name = item.decode('utf-8')
    size = int(subprocess.check_output(['git', 'cat-file', '-s', ':' + name]))
    if size >= 50 * 1024 * 1024:
        large.append((name, size))
for name, size in large:
    print(f'{size / 1024**2:.1f} MiB: {name}')
if large:
    raise SystemExit('发现大型暂存文件，先处理再提交。')
print('暂存文件大小检查通过。')
PY

git commit -m "Archive two-hop search experiments and NGFix evaluation"
```

若发现误暂存的大文件，用 `git restore --staged -- "具体路径"` 取消暂存，保留服务器文件，然后重新检查。不要删除服务器数据。

## 4. 上传到自己的私有仓库

网页已创建空仓库时：

```bash
# 把 YOUR_USERNAME 替换成自己的账号。
git remote add archive https://github.com/YOUR_USERNAME/ngfix-twohop-archive.git
git remote get-url archive
git push -u archive HEAD:refs/heads/main
```

HTTPS 身份验证使用已配置凭据、GitHub CLI 或 token，不把 token 写进仓库地址或脚本。若已经配置 GitHub SSH，可将地址替换为 `git@github.com:YOUR_USERNAME/ngfix-twohop-archive.git`。

如果网页还没建仓库，且有 GitHub CLI，则用以下命令替代上面这一组：

```bash
gh auth login
gh auth setup-git
gh repo create ngfix-twohop-archive --private --source=. --remote=archive
git push -u archive HEAD:refs/heads/main
```

若 `archive` 远程已存在，先运行 `git remote get-url archive` 核对；确认属于本次个人仓库后跳过 `git remote add`。不需要更改 origin，不需要强制推送，不需要上传所有分支或标签。

## 5. 确认已上传的版本

```bash
git rev-parse HEAD
git ls-remote archive refs/heads/main
git status --short
```

前两条输出的 commit 哈希应一致。最后一条仍可能出现未纳入归档的本地文件，应按需另存。以后在当前归档分支补充提交，用 `git push archive HEAD:refs/heads/main` 明确目标。

GitHub 普通仓库阻止超过 100 MiB 的文件；当前暂存检查采用更保守的 50 MiB 门槛。若推送仍报告旧历史中的大文件，单改忽略文件无法消除旧提交里的文件，需要另行整理历史或建立干净快照仓库，不使用强制推送试错。

参考：[GitHub CLI 创建仓库](https://cli.github.com/manual/gh_repo_create)、[GitHub 文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
