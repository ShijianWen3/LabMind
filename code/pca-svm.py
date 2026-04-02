"""
光谱数据分层分类 —— PCA + SVM 完整流程（层次分类器版本）
==========================================================
分类逻辑:
    第一阶段 (former): 合格 vs 不合格  →  class_D  vs  {class_A, class_B, class_C}
    第二阶段 (latter): 不合格细分      →  class_A  vs  class_B  vs  class_C

目录结构要求:
    train/
    ├── class_A/   *.csv
    ├── class_B/   *.csv
    ├── class_C/   *.csv
    └── class_D/   *.csv

每个 csv 文件: 两列 (x, y) 或 单列 (y)，约 8000 行
"""

import os
import sys
import glob
import joblib
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

# ══════════════════════════════════════════════
# 日志工具：同时输出到终端和文件
# ══════════════════════════════════════════════
class _Tee:
    """
    将 stdout / stderr 同时写入终端和日志文件。
    用法:
        tee = _Tee(log_path)
        ...（训练过程）...
        tee.close()
    """
    def __init__(self, log_path: str):
        self._log  = open(log_path, "w", encoding="utf-8")
        self._stdout_orig = sys.stdout
        self._stderr_orig = sys.stderr
        sys.stdout = self
        sys.stderr = self
        # 写入文件头
        self._log.write(f"{'=' * 60}\n")
        self._log.write(f"训练日志  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log.write(f"{'=' * 60}\n\n")
        self._log.flush()

    def write(self, text: str):
        self._stdout_orig.write(text)   # 正常输出到终端
        self._log.write(text)           # 同步写入文件
        self._log.flush()

    def flush(self):
        self._stdout_orig.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout_orig
        sys.stderr = self._stderr_orig
        self._log.write(f"\n{'=' * 60}\n")
        self._log.write(f"日志结束  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log.write(f"{'=' * 60}\n")
        self._log.close()
        print(f"📝 日志已保存至: {self._log.name}")


# ─────────────────────────────────────────────
# 0. 配置区  ← 根据实际情况修改这里
# ─────────────────────────────────────────────
DATA_ROOT    = "./dataset/train"   # 数据根目录
MODEL_PATH   = "./models/pca-svm"            # 模型保存根路径
TEST_SIZE    = 0.30                           # 测试集比例
RANDOM_SEED  = 42
PCA_VARIANCE = 0.95                          # PCA 保留方差比例

# 合格类别名称（与文件夹名一致）
QUALIFIED_CLASS = "class_D"
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
    返回 X (N, L)、原始标签字符串 y_str (N,) 和统一后的目标长度
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
                y_vals = df.iloc[:, -1].values.astype(float)
                X_list.append(y_vals)
                y_list.append(cls)
            except Exception as e:
                print(f"  ⚠️  读取失败 {fpath}: {e}")

        print(f"  ✅  {cls}: 加载了 {len(csv_files)} 条样本")

    if not X_list:
        raise ValueError("没有成功加载任何样本，请检查数据路径和格式")

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
    对每条光谱单独做 Min-Max 归一化（行归一化）
    消除样本间整体强度差异，保留波动形状特征
    """
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    denom = X_max - X_min
    denom[denom == 0] = 1.0
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
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED
        ))
    ])
    return pipeline


# ══════════════════════════════════════════════
# 5. 超参数搜索（可选，耗时较长）
# ══════════════════════════════════════════════
def tune_hyperparams(pipeline: Pipeline, X_train, y_train, tag: str = "") -> Pipeline:
    """
    用 GridSearchCV 搜索 SVM 的 C 和 gamma
    """
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring="f1_macro",
        n_jobs=-1, verbose=1
    )

    print(f"\n🔍 [{tag}] 正在进行超参数搜索（GridSearchCV）...")
    grid.fit(X_train, y_train)
    print(f"✅ [{tag}] 最佳参数: {grid.best_params_}")
    print(f"✅ [{tag}] 交叉验证 F1-macro: {grid.best_score_:.4f}")

    return grid.best_estimator_


# ══════════════════════════════════════════════
# 6. 评估 & 可视化
# ══════════════════════════════════════════════
def evaluate(model, X_test, y_test, class_names: list, save_dir: str, tag: str = ""):
    print(f"\n{'=' * 50}")
    print(f"Test Set Evaluation  [{tag}]")
    print("=" * 50)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    encoded_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=encoded_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix [{tag}]")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"confusion_matrix_{tag}.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to {cm_path}")


def plot_pca_variance(model, save_dir: str, tag: str = ""):
    """绘制 PCA 累积解释方差曲线"""
    pca = model.named_steps["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = pca.n_components_

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o", markersize=3, linewidth=1.5)
    plt.axhline(PCA_VARIANCE, color="red", linestyle="--",
                label=f"{PCA_VARIANCE * 100:.0f}% variance threshold")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Cumulative Variance [{tag}]  ({n_comp} components retained)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_path = os.path.join(save_dir, f"pca_variance_{tag}.png")
    plt.savefig(pca_path, dpi=150)
    plt.show()
    print(f"[{tag}] PCA: 原始维度 -> {n_comp}-dim | 方差图保存至 {pca_path}")


# ══════════════════════════════════════════════
# 7. 模型保存 / 加载
# ══════════════════════════════════════════════
def _make_timestamped_dir(base_path: str) -> str:
    """在 base_path 下创建以时间命名的子目录，返回该路径（末尾带 /）"""
    current = datetime.now()
    now_date = str(current.date())
    now_time = (str(current.time()).split(":")[0] + "-"
                + str(current.time()).split(":")[1])
    save_dir = os.path.join(base_path, f"{now_date}_{now_time}")
    save_dir += "/"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_model(model, label_encoder, target_len: int,
               save_dir: str, suffix: str) -> str:
    """
    将模型保存到 save_dir 目录下，文件名为 svm_model_{suffix}.joblib
    suffix: "former"（第一阶段）或 "latter"（第二阶段）
    返回完整文件路径
    """
    filename = f"svm_model_{suffix}.joblib"
    filepath = os.path.join(save_dir, filename)

    payload = {
        "model":         model,
        "label_encoder": label_encoder,
        "target_len":    target_len,
    }
    joblib.dump(payload, filepath)
    print(f"\n💾 [{suffix}] 模型已保存至: {filepath}  (target_len={target_len})")
    return filepath


def load_model(path: str):
    payload = joblib.load(path)
    print(f"\n📂 模型已从 {path} 加载  (target_len={payload['target_len']})")
    return payload["model"], payload["label_encoder"], payload["target_len"]


# ══════════════════════════════════════════════
# 8. 预测新样本（推理接口）
# ══════════════════════════════════════════════
def predict_single(csv_path: str, former_model_path: str, latter_model_path: str):
    """
    层次推理:
        Step1: 用 former 模型判断合格 / 不合格
        Step2: 若不合格，再用 latter 模型细分 A / B / C
    """
    model1, le1, target_len1 = load_model(former_model_path)
    model2, le2, _           = load_model(latter_model_path)

    # 读取 & 预处理
    df = pd.read_csv(csv_path, header=None)
    y_vals = df.iloc[:, -1].values.astype(float)
    y_r = resample_spectrum(y_vals, target_len1).reshape(1, -1)
    x_min, x_max = y_r.min(), y_r.max()
    y_norm = (y_r - x_min) / (x_max - x_min + 1e-8)

    # 第一阶段：合格 vs 不合格
    pred1_encoded = model1.predict(y_norm)
    pred1_proba   = model1.predict_proba(y_norm)[0]
    pred1_label   = le1.inverse_transform(pred1_encoded)[0]

    print(f"\n[former] 预测结果: {pred1_label}")
    print("[former] 类别概率:")
    for cls, prob in zip(le1.classes_, pred1_proba):
        print(f"  {cls}: {prob:.4f}")

    if pred1_label == "qualified":
        print(f"\n✅ 最终预测: {QUALIFIED_CLASS}（合格）")
        return QUALIFIED_CLASS

    # 第二阶段：A / B / C 细分
    pred2_encoded = model2.predict(y_norm)
    pred2_proba   = model2.predict_proba(y_norm)[0]
    pred2_label   = le2.inverse_transform(pred2_encoded)[0]

    print(f"\n[latter] 预测结果: {pred2_label}")
    print("[latter] 类别概率:")
    for cls, prob in zip(le2.classes_, pred2_proba):
        print(f"  {cls}: {prob:.4f}")

    print(f"\n✅ 最终预测: {pred2_label}（不合格）")
    return pred2_label


# ══════════════════════════════════════════════
# 9. 主流程
# ══════════════════════════════════════════════
def main(tune: bool = False):
    """
    tune=True  : 两阶段均执行 GridSearchCV 超参数搜索（耗时但更优）
    tune=False : 使用默认参数快速训练
    """

    # ── 加载 & 预处理 ──────────────────────────
    X_raw, y_str, target_len = load_spectra_from_folders(DATA_ROOT)
    X = preprocess(X_raw)

    # ── 划分训练/测试集（按原始4类分层，保证各类比例一致）──
    # 注意：这里用原始4分类标签做 stratify，保证两个阶段使用完全相同的划分
    le_raw = LabelEncoder()
    y_raw  = le_raw.fit_transform(y_str)

    (X_train, X_test,
     y_str_train, y_str_test) = train_test_split(
        X, y_str,
        test_size=TEST_SIZE,
        stratify=y_raw,
        random_state=RANDOM_SEED
    )

    # ── 创建时间戳目录（两个模型共用同一目录）──────
    save_dir = _make_timestamped_dir(MODEL_PATH)
    log_path = os.path.join(save_dir, "log.txt")
    tee = _Tee(log_path)                          # ← 从此处起所有 print 同步写入日志
    print(f"\n📁 本次模型保存目录: {save_dir}")
    print(f"📝 日志实时写入: {log_path}")
    print(f"测试集比例:{TEST_SIZE},Random-Seed:{RANDOM_SEED}")
    print(f"\n[total/former]训练集: {X_train.shape[0]} 条 | 测试集: {X_test.shape[0]} 条")
    
    

    # ══════════════════════════════════════════
    # 第一阶段（former）: 合格 vs 不合格
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  第一阶段 [former]: 合格 vs 不合格")
    print("=" * 60)

    # 将 A/B/C 合并为 "defective"，D 映射为 "qualified"
    y_binary_train = np.where(y_str_train == QUALIFIED_CLASS, "qualified", "defective")
    y_binary_test  = np.where(y_str_test  == QUALIFIED_CLASS, "qualified", "defective")


    # 打印训练集各类样本数
    print("\n[total/former] 训练集类别分布:")
    for cls in np.unique(y_str_train):
        print(f"  {cls}: {np.sum(y_str_train == cls)} 条")

    # 打印合格/不合格分布（former 视角）
    qualified_tr = np.sum(y_str_train == QUALIFIED_CLASS)
    defective_tr = np.sum(y_str_train != QUALIFIED_CLASS)
    print(f"\n[former] 训练集正负类分布:")
    print(f"  qualified ({QUALIFIED_CLASS}): {qualified_tr} 条")
    print(f"  defective (A+B+C) : {defective_tr} 条")

    le1 = LabelEncoder()
    y_tr1 = le1.fit_transform(y_binary_train)
    y_te1 = le1.transform(y_binary_test)

    pipeline1 = build_pipeline()
    if tune:
        model1 = tune_hyperparams(pipeline1, X_train, y_tr1, tag="former")
    else:
        print("\n🚀 [former] 开始训练（默认参数）...")
        pipeline1.fit(X_train, y_tr1)
        model1 = pipeline1
        print("✅ [former] 训练完成")

    # 保存第一阶段模型
    save_model(model1, le1, target_len, save_dir, suffix="former")

    # 评估第一阶段
    plot_pca_variance(model1, save_dir, tag="former")
    evaluate(model1, X_test, y_te1, list(le1.classes_), save_dir, tag="former")

    # ══════════════════════════════════════════
    # 第二阶段（latter）: 不合格细分 A / B / C
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  第二阶段 [latter]: 不合格细分 A / B / C")
    print("=" * 60)

    # 从训练集和测试集中各自筛出不合格样本
    mask_train_defect = (y_str_train != QUALIFIED_CLASS)
    mask_test_defect  = (y_str_test  != QUALIFIED_CLASS)

    X_tr2      = X_train[mask_train_defect]
    y_str_tr2  = y_str_train[mask_train_defect]
    X_te2      = X_test[mask_test_defect]
    y_str_te2  = y_str_test[mask_test_defect]

    print(f"\n[latter] 训练样本: {X_tr2.shape[0]} 条 | 测试样本: {X_te2.shape[0]} 条")
    for cls in np.unique(y_str_tr2):
        print(f"  训练集 {cls}: {np.sum(y_str_tr2 == cls)} 个样本")

    le2 = LabelEncoder()
    y_tr2 = le2.fit_transform(y_str_tr2)
    y_te2 = le2.transform(y_str_te2)

    pipeline2 = build_pipeline()
    if tune:
        model2 = tune_hyperparams(pipeline2, X_tr2, y_tr2, tag="latter")
    else:
        print("\n🚀 [latter] 开始训练（默认参数）...")
        pipeline2.fit(X_tr2, y_tr2)
        model2 = pipeline2
        print("✅ [latter] 训练完成")

    # 保存第二阶段模型
    save_model(model2, le2, target_len, save_dir, suffix="latter")

    # 评估第二阶段
    plot_pca_variance(model2, save_dir, tag="latter")
    evaluate(model2, X_te2, y_te2, list(le2.classes_), save_dir, tag="latter")

    # ══════════════════════════════════════════
    # 端到端联合评估（在完整测试集上跑层次推理）
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  端到端联合评估（层次推理 on 完整测试集）")
    print("=" * 60)

    # 用 former 模型对全部测试样本判断合格/不合格
    X_test_norm  = X_test
    y_pred_final = []

    pred1_all = le1.inverse_transform(model1.predict(X_test_norm))

    for i, pred1 in enumerate(pred1_all):
        if pred1 == "qualified":
            y_pred_final.append(QUALIFIED_CLASS)
        else:
            # 送入 latter 模型细分
            pred2_encoded = model2.predict(X_test_norm[i:i+1])
            pred2_label   = le2.inverse_transform(pred2_encoded)[0]
            y_pred_final.append(pred2_label)

    y_pred_final = np.array(y_pred_final)

    # 用原始4类标签做最终评估
    all_classes = sorted(np.unique(y_str_test).tolist())
    le_final = LabelEncoder().fit(all_classes)
    y_te_enc  = le_final.transform(y_str_test)
    y_pred_enc = le_final.transform(y_pred_final)

    print("\nClassification Report (端到端层次推理):")
    print(classification_report(y_te_enc, y_pred_enc, target_names=all_classes))

    cm = confusion_matrix(y_te_enc, y_pred_enc, labels=list(range(len(all_classes))))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix [End-to-End Hierarchical]")
    plt.tight_layout()
    e2e_path = os.path.join(save_dir, "confusion_matrix_e2e.png")
    plt.savefig(e2e_path, dpi=150)
    plt.show()
    print(f"端到端混淆矩阵保存至: {e2e_path}")

    print(f"\n🎉 全部完成！模型和图表保存在: {save_dir}")
    print(f"   ├── svm_model_former.joblib   （第一阶段：合格/不合格）")
    print(f"   ├── svm_model_latter.joblib   （第二阶段：A/B/C 细分）")
    print(f"   ├── pca_variance_former.png")
    print(f"   ├── pca_variance_latter.png")
    print(f"   ├── confusion_matrix_former.png")
    print(f"   ├── confusion_matrix_latter.png")
    print(f"   ├── confusion_matrix_e2e.png")
    print(f"   └── log.txt")

    tee.close()                                   # ← 关闭日志，恢复 stdout/stderr
    return save_dir


# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 正常训练：main(tune=False)
    # 超参搜索：main(tune=True)
    main(tune=False)

    # 推理示例（取消注释使用，路径改为实际保存路径）:
    # predict_single(
    #     csv_path       = "./dataset/2026-3-24/test/sample.csv",
    #     former_model_path = "./models/pca-svm/2026-03-24_10-30/svm_model_former.joblib",
    #     latter_model_path = "./models/pca-svm/2026-03-24_10-30/svm_model_latter.joblib",
    # )