import yaml
from model.trainer import train
from utils.plotting import plot_learning_curve, plot_confusion_matrix

with open("config/params.yml") as f:
    config = yaml.safe_load(f)

model, history, cm, metrics = train(config)

plot_learning_curve(history, save_path="results/plots/learning_curve.png")
plot_confusion_matrix(cm, save_path="results/plots/confusion_matrix.png")

print("\nEvaluation metrics:", metrics)
