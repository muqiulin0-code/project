# 推送到 GitHub

## 方式 A：使用 VS Code（推荐）

1. 在 VS Code 打开本目录。
2. 打开 Source Control，点击 `Publish Branch`。
3. 如果已经登录 GitHub，选择账号、仓库名和 Public/Private。
4. 首次发布后即可按需提交并点击 `Sync Changes`。

## 方式 B：先手动创建空仓库，再使用命令

在 GitHub 网页创建一个**不要初始化 README**的空仓库，然后执行：

```bash
git remote add origin git@github.com:<用户名>/<仓库名>.git
git push -u origin main
```

如果使用 HTTPS：

```bash
git remote add origin https://github.com/<用户名>/<仓库名>.git
git push -u origin main
```

VS Code 的 GitHub 登录可以用于 HTTPS 推送，但 Git 本身仍需要正确配置用户名和邮箱：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的 GitHub 邮箱"
```

注意：不要提交 `.venv/`、CIFAR 数据集或大体积模型权重；本仓库的 `.gitignore` 已处理这些文件。
