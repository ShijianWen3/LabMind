"""
光谱数据分类 —— PCA + SVM 完整流程
=====================================
目录结构要求:
    data/
    ├── class_A/   *.csv
    ├── class_B/   *.csv
    ├── class_C/   *.csv
    └── class_D/   *.csv

每个 csv 文件: 两列 (x, y) 或 单列 (y)，约 8000 行
"""

import os
import glob
import joblib
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# 0. 配置区  ← 根据实际情况修改这里
# ─────────────────────────────────────────────
DATA_ROOT   = "./dataset/2026-3-18/train"          # 数据根目录
MODEL_PATH  = "./models/pca-svm"  # 模型保存路径
TEST_SIZE   = 0.4               # 测试集比例
RANDOM_SEED = 33
PCA_VARIANCE= 0.95              # PCA 保留方差比例
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# 1. 工具函数：插值重采样
# ══════════════════════════════════════════════
def resample_spectrum(y_vals: np.ndarray, target_len: int) -> np.ndarray:
    """
    将任意长度的光谱插值重采样到 target_len 个点。
    使用线性插值，保留整体波形形状，不丢弃任何区段。
    """
    if len(y_vals) == target_len:
        return y_vals
    x_old = np.linspace(0, 1, len(y_vals))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, y_vals, kind="linear")
    return f(x_new)


# ══════════════════════════════════════════════
# 2. 数据加载
# ══════════════════════════════════════════════
def load_spectra_from_folders(root: str):
    """
    遍历 root 下的每个子文件夹作为一个类别，
    读取文件夹内所有 csv，只取 y 列（最后一列）作为特征向量。
    返回 X (N, L) 和 labels (N,)
    """
    X_list, y_list = [], []

    class_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    if not class_dirs:
        raise FileNotFoundError(f"在 {root} 下未找到任何子文件夹")

    print(f"发现 {len(class_dirs)} 个类别: {class_dirs}\n")

    for cls in class_dirs:
        csv_files = glob.glob(os.path.join(root, cls, "*.csv"))
        if not csv_files:
            print(f"  ⚠️  {cls} 文件夹下没有 csv 文件，跳过")
            continue

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath, header=None)
                # 如果是两列(x,y)就取最后一列；单列直接取
                y_vals = df.iloc[:, -1].values.astype(float)
                X_list.append(y_vals)
                y_list.append(cls)
            except Exception as e:
                print(f"  ⚠️  读取失败 {fpath}: {e}")

        print(f"  ✅  {cls}: 加载了 {len(csv_files)} 条样本")

    if not X_list:
        raise ValueError("没有成功加载任何样本，请检查数据路径和格式")

    # 统一长度：插值重采样到所有样本的中位数长度
    # 比截断更好：保留完整波形，不丢弃任何区段的信息
    lengths = [len(x) for x in X_list]
    target_len = int(np.median(lengths))
    print(f"\n各样本长度范围: {min(lengths)} ~ {max(lengths)}，统一插值到 {target_len} 点")

    X = np.array([resample_spectrum(x, target_len) for x in X_list])
    y = np.array(y_list)

    print(f"\n数据加载完成: X={X.shape}, 类别分布:")
    for cls in np.unique(y):
        print(f"  {cls}: {np.sum(y == cls)} 个样本")

    return X, y, target_len


# ══════════════════════════════════════════════
# 3. 预处理
# ══════════════════════════════════════════════
def preprocess(X: np.ndarray) -> np.ndarray:
    """
    对每条光谱单独做 Min-Max 归一化 (行归一化)
    消除样本间整体强度差异，保留波动形状特征
    """
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    denom = X_max - X_min
    denom[denom == 0] = 1.0          # 防止除零
    X_norm = (X - X_min) / denom
    print(f"\n✅ Min-Max 归一化完成, shape: {X_norm.shape}")
    return X_norm


# ══════════════════════════════════════════════
# 4. 构建 Pipeline: StandardScaler → PCA → SVM
# ══════════════════════════════════════════════
def build_pipeline(n_components_ratio: float = PCA_VARIANCE) -> Pipeline:
    """
    StandardScaler: 列方向标准化（PCA 的前置要求）
    PCA:            降维，保留指定方差比例
    SVM:            RBF 核，支持类别不均衡
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components_ratio, random_state=RANDOM_SEED)),
        ("svm",    SVC(
            kernel="rbf",
            class_weight="balanced",   # ← 自动处理类别不均衡
            probability=True,          # 支持 predict_proba
            random_state=RANDOM_SEED
        ))
    ])
    return pipeline


# ══════════════════════════════════════════════
# 4. 超参数搜索（可选，耗时较长）
# ══════════════════════════════════════════════
def tune_hyperparams(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    用 GridSearchCV 搜索 SVM 的 C 和 gamma
    如果样本量大可以缩小搜索范围节省时间
    """
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring="f1_macro",       # 适合不均衡数据
        n_jobs=-1, verbose=1
    )

    print("\n🔍 正在进行超参数搜索（GridSearchCV）...")
    grid.fit(X_train, y_train)
    print(f"✅ 最佳参数: {grid.best_params_}")
    print(f"✅ 交叉验证 F1-macro: {grid.best_score_:.4f}")

    return grid.best_estimator_


# ══════════════════════════════════════════════
# 5. 评估 & 可视化
# ══════════════════════════════════════════════
def evaluate(model, X_test, y_test, class_names, path:str):
    print("\n" + "=" * 50)
    print("Test Set Evaluation")
    print("=" * 50)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 修复：labels 传编码后的整数，display_labels 传类名字符串
    encoded_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=encoded_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(path+"confusion_matrix.png", dpi=150)
    plt.show()
    print("\nConfusion matrix saved to confusion_matrix.png")


def plot_pca_variance(model, path:str):
    """Plot cumulative explained variance of PCA components."""
    pca = model.named_steps["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = pca.n_components_

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o", markersize=3, linewidth=1.5)
    plt.axhline(PCA_VARIANCE, color="red", linestyle="--",
                label=f"{PCA_VARIANCE*100:.0f}% variance threshold")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Cumulative Variance  ({n_comp} components retained)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path+"pca_variance.png", dpi=150)
    plt.show()
    print(f"PCA: 8000-dim -> {n_comp}-dim | variance plot saved to pca_variance.png")


# ══════════════════════════════════════════════
# 7. 模型保存 / 加载
# ══════════════════════════════════════════════
def save_model(model, label_encoder, target_len: int, path: str = MODEL_PATH):

    current = datetime.now()
    now_date = str(current.date())
    now_time = str(current.time()).split(":")[0] + "-" +str(current.time()).split(":")[1]
    path = path + "/" + now_date + "_" + now_time + "/"

    path_result = copy.copy(path)

    #创建路径
    os.makedirs(os.path.dirname(path), exist_ok=True)

    path = path + "svm_model.joblib"


    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "target_len": target_len,   # ← 保存训练时的目标长度，推理时用于对齐
    }
    joblib.dump(payload, path)
    print(f"\n💾 模型已保存至: {path}  (target_len={target_len})")

    return path_result


def load_model(path: str = MODEL_PATH):
    payload = joblib.load(path)
    print(f"\n📂 模型已从 {path} 加载  (target_len={payload['target_len']})")
    return payload["model"], payload["label_encoder"], payload["target_len"]


# ══════════════════════════════════════════════
# 8. 预测新样本（推理接口）
# ══════════════════════════════════════════════
def predict_single(csv_path: str, model_path: str = MODEL_PATH):
    """
    加载保存的模型，对单个 csv 文件进行预测。
    自动将输入光谱插值对齐到训练时的目标长度。
    """
    model, le, target_len = load_model(model_path)

    df = pd.read_csv(csv_path, header=None)
    y_vals = df.iloc[:, -1].values.astype(float)

    # ── 关键：插值对齐到训练时的长度 ──────────────
    y_resampled = resample_spectrum(y_vals, target_len).reshape(1, -1)
    print(f"输入长度: {len(y_vals)} → 对齐到: {target_len}")

    # Min-Max 归一化（与训练时一致）
    x_min, x_max = y_resampled.min(), y_resampled.max()
    y_norm = (y_resampled - x_min) / (x_max - x_min + 1e-8)

    pred_encoded = model.predict(y_norm)
    pred_proba   = model.predict_proba(y_norm)
    pred_label   = le.inverse_transform(pred_encoded)[0]

    print(f"\nPrediction: {pred_label}")
    print("Class probabilities:")
    for cls, prob in zip(le.classes_, pred_proba[0]):
        print(f"  {cls}: {prob:.4f}")

    return pred_label


# ══════════════════════════════════════════════
# 9. 主流程
# ══════════════════════════════════════════════
def main(tune: bool = False):
    """
    tune=True  : 执行 GridSearchCV 超参数搜索（耗时但更优）
    tune=False : 使用默认参数快速训练
    """

    # ── 加载 & 预处理 ──────────────────────────
    X_raw, y_str, target_len = load_spectra_from_folders(DATA_ROOT)
    X = preprocess(X_raw)

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = list(le.classes_)

    # ── 划分训练/测试集 ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )
    print(f"\n训练集: {X_train.shape[0]} 条 | 测试集: {X_test.shape[0]} 条")

    # ── 构建 & 训练 ─────────────────────────────
    pipeline = build_pipeline()

    if tune:
        model = tune_hyperparams(pipeline, X_train, y_train)
    else:
        print("\n🚀 开始训练（默认参数）...")
        pipeline.fit(X_train, y_train)
        model = pipeline
        print("✅ 训练完成")

    # ── 保存模型（含 target_len）────────────────
    path = save_model(model, le, target_len)

    # ── PCA 信息 ────────────────────────────────
    plot_pca_variance(model, path)

    # ── 测试评估 ────────────────────────────────
    evaluate(model, X_test, y_test, class_names, path)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 正常训练：main(tune=False)
    # 超参搜索：main(tune=True)
    main(tune=True)

    # 推理示例（取消注释使用）:
    # predict_single("./dataset/2026-3-18/train/class_A/2.csv", "./models/pca-svm/svm_model-2026-03-19_19-20.joblib")
