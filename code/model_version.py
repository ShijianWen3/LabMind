import joblib
import sklearn

print("当前 sklearn 版本:", sklearn.__version__)

m1 = joblib.load("./models/pca-svm/2026-04-02_16-39/svm_model_former.joblib")
m2 = joblib.load("./models/pca-svm/2026-04-02_16-39/svm_model_latter.joblib")

# 查看模型对象的版本记录
for name, payload in [("former", m1), ("latter", m2)]:
    model = payload.get("model")
    print(f"\n{name} model type:", type(model))
    if hasattr(model, '__sklearn_version__'):
        print(f"{name} sklearn version:", model.__sklearn_version__)
    # Pipeline 的话看 steps
    if hasattr(model, 'steps'):
        for step_name, step in model.steps:
            print(f"  step '{step_name}':", type(step).__name__, 
                  getattr(step, '__sklearn_version__', 'N/A'))