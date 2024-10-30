import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Epoch": list(range(1, 11)),
    "ResNet18": [0.2591, 0.3839, 0.4518, 0.4534, 0.4645, 0.5055, 0.5166, 0.5308, 0.5229, 0.5261],
    "ResNet50": [0.2528, 0.3476, 0.4487, 0.4423, 0.4676, 0.4882, 0.4866, 0.4897,0.5039,0.5182],
    "MobileNetV2_LR_001": [0.2749, 0.4028, 0.4787, 0.5055, 0.5355, 0.5371, 0.5624, 0.5577, 0.5829, 0.5798],
    "MobileNetV2_LR_003": [0.3049, 0.4455, 0.5008, 0.5103, 0.5150, 0.5482, 0.5403, 0.5482, 0.5450, 0.5608],
    "MobileNetV3_Large": [0.3333, 0.4945, 0.5671, 0.5782, 0.5687, 0.5814, 0.5719, 0.5829, 0.5877, 0.5861],
    "MobileNetV3_Small": [0.2686, 0.4123, 0.4487, 0.4676, 0.4834, 0.4913, 0.4929, 0.5039, 0.5118, 0.5118],
    "EfficientNet": [0.3017, 0.4013, 0.4471, 0.4613, 0.4771, 0.5087, 0.5071, 0.5055, 0.5213, 0.5229]
}


df = pd.DataFrame(data)

plt.figure(figsize=(12, 8))
for column in df.columns[1:]:
    plt.plot(df["Epoch"], df[column], label=column)

plt.ylim(0.25, 0.65)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison across Models")
plt.legend(loc="lower right", fontsize=12)
plt.grid(visible=True)

plt.show()