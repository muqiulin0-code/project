# 同步到 GitHub

当前仓库已经配置远程地址：

```text
origin  https://github.com/muqiulin0-code/project.git
branch  main
```

## 推荐流程：VS Code

1. 在 VS Code 打开本项目。
2. 确认 Source Control 右上角已经显示 GitHub 账号 `muqiulin0-code`。
3. 检查 Changes，确认没有提交数据集、模型权重、`.venv/` 或临时锁文件。
4. 填写提交信息并点击 **Commit**。
5. 点击 **Sync Changes**，本地 `main` 就会同步到 `origin/main`。

## 命令行流程

```bash
conda activate pytorch
make check

git status
git add .
git commit -m "chore: use shared conda pytorch environment"
git push origin main
```

Git 本身不会自动复用 VS Code 扩展里的登录状态；如果命令行推送要求认证，优先在 VS Code 的 Source Control 中点击 **Sync Changes**，或在第一次推送时完成 GitHub 登录。

不要提交以下内容，本仓库的 `.gitignore` 已处理：

- `.venv/`
- `data/` 和 CIFAR-10 数据集
- `models/`
- `artifacts/cifar10/` 中的模型权重
- `.vscode/` 中的本地数据库和缓存
