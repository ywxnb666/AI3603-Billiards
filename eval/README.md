# Eval

本目录放置**测试所需**文件：checkpoint 等。

## 文件
- `muzero_sklearn.joblib`: `NewAgent` 默认加载的模型文件

## 对战命令
在项目根目录运行：

```bash
python evaluate.py
```

`evaluate.py` 会让 `BasicAgent()` 与 `NewAgent()` 对战，并输出 120 局统计结果。
